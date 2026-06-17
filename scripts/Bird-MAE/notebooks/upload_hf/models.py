import math
from typing import Dict, List, Optional, Tuple
import warnings

import torch
from torch import nn
import lightning as L
import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed, VisionTransformer
from timm.models.layers import trunc_normal_
from util.pos_embed import get_2d_sincos_pos_embed_flexible
from util.patch_embed import PatchEmbed_new
from models_ppnet import LinearLayerWithoutNegativeConnections, PPNet

class NonNegativeLinear(nn.Module):
    """
    A PyTorch module for a linear layer with non-negative weights.

    This module applies a linear transformation to the incoming data: `y = xA^T + b`.
    The weights of the transformation are constrained to be non-negative, making this
    module particularly useful in models where negative weights may not be appropriate.

    Attributes:
        in_features (int): The number of features in the input tensor.
        out_features (int): The number of features in the output tensor.
        weight (torch.Tensor): The weight parameter of the module, constrained to be non-negative.
        bias (torch.Tensor, optional): The bias parameter of the module.

    Args:
        in_features (int): The number of features in the input tensor.
        out_features (int): The number of features in the output tensor.
        bias (bool, optional): If True, the layer will include a learnable bias. Default: True.
        device (optional): The device (CPU/GPU) on which to perform computations.
        dtype (optional): The data type for the parameters (e.g., float32).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the NonNegativeLinear module.

        Args:
            input (torch.Tensor): The input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_features).
        """
        return nn.functional.linear(input, torch.relu(self.weight), self.bias)


class LinearLayerWithoutNegativeConnections(nn.Module):
    r"""
    Custom Linear Layer where each output class is connected to a specific subset of input features.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
        device: the device of the module parameters. Default: ``None``
        dtype: the data type of the module parameters. Default: ``None``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out_features}, \text{features_per_output_class})`.
        bias: the learnable bias of the module of shape :math:`(\text{out_features})`.
              If :attr:`bias` is ``True``, the values are initialized from
              :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
              :math:`k = \frac{1}{\text{features_per_output_class}}`
    """

    __constants__ = ["in_features", "out_features", "bias"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        non_negative: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.non_negative = non_negative

        # Calculate the number of features per output class
        self.features_per_output_class = in_features // out_features

        print(self.in_features, self.out_features, self.features_per_output_class)

        # Ensure input size is divisible by the output size
        assert (
            in_features % out_features == 0
        ), "in_features must be divisible by out_features"

        # Define weights and biases
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, self.features_per_output_class), **factory_kwargs
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Initialize weights and biases
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize the weights and biases.
        Weights are initialized using Kaiming uniform initialization.
        Biases are initialized using a uniform distribution.
        """
        # Kaiming uniform initialization for the weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            # Calculate fan-in and fan-out values
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)

            # Uniform initialization for the biases
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        batch_size = input.size(0)
        # Reshape input to (batch_size, out_features, features_per_output_class)
        reshaped_input = input.view(
            batch_size, self.out_features, self.features_per_output_class
        )

        # Apply ReLU to weights if non_negative_last_layer is True
        weight = torch.relu(self.weight) if self.non_negative else self.weight

        # Perform batch matrix multiplication and add bias
        output = torch.einsum("bof,of->bo", reshaped_input, weight)

        if self.bias is not None:
            output += self.bias

        return output

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class PPNet(nn.Module):
    def __init__(
        self,
        num_prototypes: 2055,
        channels_prototypes: 1024,
        h_prototypes: 1, 
        w_prototypes: 1,
        num_classes: int,
        topk_k: int = 1,
        margin: Optional[float] = None,
        init_weights: bool = True,
        add_on_layers_type: str = "bottleneck",
        incorrect_class_connection: Optional[float] = -0.5,
        correct_class_connection: float = 1.0,
        bias_last_layer: Optional[float] = None,
        non_negative_last_layer: bool = True,
        embedded_spectrogram_height: Optional[int] = None,
    ) -> None:
        """
        PPNet is a class that implements the Prototypical Part Network (ProtoPNet) for prototype-based classification.

        Args:
            backbone_model (nn.Module): A PyTorch model to use as a base architecture for the PPNet.
            prototype_shape (Tuple[int, int, int, int]): A tuple representing the shape of the prototypes in the PPNet,
                as (number of prototypes, number of channels, prototype height, prototype width).
            num_classes (int): The number of classes in the classification task.
            topk_k (int): The number of top prototype activations to consider during training (default=1).
            margin (Optional[float]): The margin to use for subtractive margin cross entropy. (default=None).
            init_weights (bool): A boolean indicating whether to initialize the weights of the PPNet (default=True).
            add_on_layers_type (str): A string indicating the type of additional layers to add to the base architecture,
                can be 'bottleneck', 'identity', or 'upsample' (default='bottleneck').
            incorrect_class_connection (float): A float value incorrect class connections are initialized to
            (default=-1).

         Raises:
            ValueError: Raises an error if the number of prototypes is not evenly divisible by the number
             of classes, i.e. there are not the same number of prototypes for each class.
        """

        super().__init__()
        #change so that it must not be calculated before
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes * self.num_classes        
        self.prototype_shape = (
            self.num_prototypes,
            channels_prototypes,
            h_prototypes,
            w_prototypes,
        )
        #self.num_prototypes = num_prototypes
        self.num_prototypes_after_pruning = None
        self.margin = margin
        self.relu_on_cos = True
        self.incorrect_class_connection = incorrect_class_connection
        self.correct_class_connection = correct_class_connection
        self.input_vector_length = 64
        self.n_eps_channels = 2
        self.epsilon_val = 1e-4
        self.topk_k = topk_k
        self.bias_last_layer = bias_last_layer
        self.non_negative_last_layer = non_negative_last_layer
        self.embedded_spectrogram_height = embedded_spectrogram_height

        if self.bias_last_layer:
            self.use_bias_last_layer = True
        else:
            self.use_bias_last_layer = False

        self.prototype_class_identity = None
        self.num_prototypes_per_class = None

        # Checking the number of prototypes
        if self.num_prototypes % self.num_classes != 0:
            warnings.warn(
                "Number of prototypes is not evenly divisible by the number of classes.",
                UserWarning,
            )
        else:
            # Calculate the number of prototypes per class
            self.num_prototypes_per_class = self.num_prototypes // self.num_classes

            # Create a 1D tensor where each element represents the class index
            self.prototype_class_identity = (
                torch.arange(self.num_prototypes) // self.num_prototypes_per_class
            )

        # for j in range(self.num_prototypes):
        #    self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1

        # this has to be named backbone_model to allow the precise loading
        #self.backbone_model = backbone_model

        self._setup_add_on_layers(add_on_layers_type=add_on_layers_type)

        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )

        if self.embedded_spectrogram_height:
            # Initialize the frequency weights with a large positive value of 3.0 so that sigmoid(frequency_weights) is close to 1.
            self.frequency_weights = nn.Parameter(
                torch.full(
                    (
                        self.num_prototypes,
                        self.embedded_spectrogram_height,
                    ),
                    3.0,
                )
            )
        else:
            self.frequency_weights = None

        if self.incorrect_class_connection:
            if self.non_negative_last_layer:
                self.last_layer = NonNegativeLinear(
                    self.num_prototypes, self.num_classes, bias=self.use_bias_last_layer
                ) # kann nur charakteristische prototypen lernen (im pipnet paper)
            else:
                self.last_layer = nn.Linear(
                    self.num_prototypes, self.num_classes, bias=self.use_bias_last_layer
                )
        else:
            self.last_layer = LinearLayerWithoutNegativeConnections(
                in_features=self.num_prototypes,
                out_features=self.num_classes,
                non_negative=self.non_negative_last_layer,
            )

        if init_weights:
            self._initialize_weights()

    def _setup_add_on_layers(self, add_on_layers_type: str):
        """
        Configures additional layers based on the backbone model architecture and the specified add_on_layers_type.

        Args:
            add_on_layers_type (str): Type of additional layers to add. Can be 'identity' or 'upsample'.
        """

        if add_on_layers_type == "identity":
            self.add_on_layers = nn.Sequential(nn.Identity())
        elif add_on_layers_type == "upsample":
            self.add_on_layers = nn.Upsample(scale_factor=2, mode="bilinear")
        else:
            raise NotImplementedError(
                f"The add-on layer type {add_on_layers_type} isn't implemented yet."
            )

    def conv_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Takes an input tensor and passes it through the backbone model to extract features.
        Then, it passes them through the additional layers to produce the output tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the backbone model and additional layers.
        """
        # Extract features using the backbone model
        #features = self.backbone_model(x)
        features = features

        # The features must be a 4D tensor of shape (batch size, channels, height, width)
        if features.dim() == 3:
            features.unsqueeze_(0)

        # Pass the features through additional layers
        output = self.add_on_layers(features) # 64, 1024, 16, 64

        return output

    def l1_activation(
            self,
            x: torch.Tensor,
            prototypes_of_wrong_class: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the L2 (Euclidean) activation between the input tensor x and the prototype vectors.
        For each patch in x and each prototype in self.prototype_vectors, we compute the squared L2 distance:
        
            d = ||x_patch||^2 + ||p||^2 - 2 * (x_patch • p)
        
        Then, we transform this distance into a similarity score as:
        
            s = log((d + 1) / (d + epsilon))
        
        Global max pooling over the spatial dimensions is applied to obtain one score per prototype.
        
        If margin adjustments are enabled (and prototypes_of_wrong_class is provided), a margin is subtracted
        for the wrong-class prototypes.
        
        Parameters:
            x : torch.Tensor
                Input tensor with shape (batch_size, num_channels, H, W).
            prototypes_of_wrong_class : Optional[torch.Tensor]
                Tensor for wrong-class prototypes (used for margin adjustments).
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - activations: The similarity scores with margin adjustments (if applicable).
                - marginless_activations: The similarity scores without margin adjustments.
        """
        # Compute squared norm for each patch in x: shape (batch_size, 1, H, W)
        x_norm_sq = torch.sum(x ** 2, dim=1, keepdim=True)
        
        # Compute squared norm for each prototype.
        # Assume self.prototype_vectors has shape (num_prototypes, num_channels, prototype_H, prototype_W)
        p_norm_sq = torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True)  # shape: (num_prototypes, 1, 1, 1)
        
        # Compute dot product between x and prototypes using convolution.
        # Resulting shape: (batch_size, num_prototypes, H, W)
        dot_product = nn.functional.conv2d(x, self.prototype_vectors)
        
        # Compute squared Euclidean distance: d = ||x||^2 + ||p||^2 - 2 * (x • p)
        d = x_norm_sq + p_norm_sq - 2 * dot_product
        d = torch.clamp(d, min=0.0)  # Ensure non-negative distances
        
        # Transform distance into similarity:
        # s = log((d + 1) / (d + epsilon)), where self.epsilon_val is a small constant to avoid division by zero.
        similarity = torch.log((d + 1.0) / (d + self.epsilon_val))
        
        # Global max pooling over spatial dimensions (H and W) to get a single similarity score per prototype.
        marginless_activations = torch.amax(similarity, dim=[2, 3])  # shape: (batch_size, num_prototypes)
        
        if self.margin is None or not self.training or prototypes_of_wrong_class is None:
            activations = marginless_activations
        else:
            # For margin adjustments, subtract a margin for wrong-class prototypes.
            # Here, we assume prototypes_of_wrong_class has shape (batch_size, num_prototypes)
            wrong_class_margin = (prototypes_of_wrong_class * self.margin).view(x.size(0),
                                                                            self.prototype_vectors.size(0),
                                                                            1, 1)
            wrong_class_margin = wrong_class_margin.expand(-1, -1, similarity.size(2), similarity.size(3))
            penalized_similarity = torch.log((d + 1.0) / (d + self.epsilon_val)) - wrong_class_margin
            activations = torch.amax(penalized_similarity, dim=[2, 3])
        
        return activations, marginless_activations

    def cos_activation(
        self,
        x: torch.Tensor,
        prototypes_of_wrong_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cosine activation between input tensor x and prototype vectors.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor with shape (batch_size, num_channels, height, width).
        prototypes_of_wrong_class : Optional[torch.Tensor]
            Tensor containing the prototypes of the wrong class with shape (batch_size, num_prototypes).

        Returns:
        --------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing:
            - activations: The cosine activations with potential margin adjustments.
            - marginless_activations: The cosine activations without margin adjustments.
        """
        input_vector_length = self.input_vector_length
        normalizing_factor = (
            self.prototype_shape[-2] * self.prototype_shape[-1]
        ) ** 0.5

        # Pre-allocate epsilon channels on the correct device for input tensor x
        epsilon_channel_x = torch.full(
            (x.shape[0], self.n_eps_channels, x.shape[2], x.shape[3]),
            self.epsilon_val,
            device=x.device,
            requires_grad=False,
        )
        x = torch.cat((x, epsilon_channel_x), dim=-3)

        # Normalize x
        x_length = torch.sqrt(torch.sum(x**2, dim=-3, keepdim=True) + self.epsilon_val)
        x_normalized = (input_vector_length * x / x_length) / normalizing_factor

        # Pre-allocate epsilon channels for prototypes on the correct device
        epsilon_channel_p = torch.full(
            (
                self.prototype_shape[0],
                self.n_eps_channels,
                self.prototype_shape[2],
                self.prototype_shape[3],
            ),
            self.epsilon_val,
            device=self.prototype_vectors.device,
            requires_grad=False,
        )
        appended_protos = torch.cat((self.prototype_vectors, epsilon_channel_p), dim=-3)

        # Normalize prototypes
        prototype_vector_length = torch.sqrt(
            torch.sum(appended_protos**2, dim=-3, keepdim=True) + self.epsilon_val
        )
        normalized_prototypes = appended_protos / (
            prototype_vector_length + self.epsilon_val
        )
        normalized_prototypes /= normalizing_factor

        # Compute activations using convolution
        activations_dot = nn.functional.conv2d(x_normalized, normalized_prototypes)
        marginless_activations = activations_dot / (input_vector_length * 1.01)

        if self.frequency_weights is not None:
            # Apply sigmoid to frequency weights. s.t. weights are between 0 and 1.
            freq_weights = torch.sigmoid(self.frequency_weights)

            # Multiply each prototype's frequency response by the corresponding weights
            marginless_activations = marginless_activations * freq_weights[:, :, None]

        if (
            self.margin is None
            or not self.training
            or prototypes_of_wrong_class is None
        ):
            activations = marginless_activations
        else:
            # Apply margin adjustment for wrong class prototypes
            wrong_class_margin = (prototypes_of_wrong_class * self.margin).view(
                x.size(0), self.prototype_vectors.size(0), 1, 1
            )
            wrong_class_margin = wrong_class_margin.expand(
                -1, -1, activations_dot.size(-2), activations_dot.size(-1)
            )
            penalized_angles = (
                torch.acos(activations_dot / (input_vector_length * 1.01))
                - wrong_class_margin
            )
            activations = torch.cos(torch.relu(penalized_angles))

        if self.relu_on_cos:
            # Apply ReLU activation on the cosine values
            activations = torch.relu(activations)
            marginless_activations = torch.relu(marginless_activations)

        return activations, marginless_activations

    def prototype_activations(
        self,
        x: torch.Tensor,
        prototypes_of_wrong_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute the prototype activations for a given input tensor.

        Args:
        - x (torch.Tensor): The raw input tensor with shape (batch_size, num_channels, height, width).
        - prototypes_of_wrong_class (Optional[torch.Tensor]): The prototypes of the wrong classes that are needed
            when using subtractive margins. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
            - activations: A tensor containing the prototype activations.
            - a list containing:
                - marginless_activations: A tensor containing the activations before applying subtractive margin.
                - conv_features: A tensor containing the convolutional features.
        """
        # Compute convolutional features
        # in: 64, 1024, 8, 32
        conv_features = self.conv_features(x)

        # Compute cosine activations
        activations, marginless_activations = self.cos_activation(
            conv_features,
            prototypes_of_wrong_class=prototypes_of_wrong_class,
        )

        return activations, [marginless_activations, conv_features]

    def forward(
        self,
        x: torch.Tensor,
        prototypes_of_wrong_class: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the PPNet model.

        Args:
        - x (torch.Tensor): Input tensor with shape (batch_size, num_channels, height, width).
        - prototypes_of_wrong_class (Optional[torch.Tensor]): The prototypes of the wrong classes that are needed
            when using subtractive margins. Defaults to None.

        Returns:
            Tuple[torch.Tensor, List[torch.Tensor]]:
            - logits: A tensor containing the logits for each class in the model.
            - a list containing:
                - mean_activations: A tensor containing the mean of the top-k prototype activations.
                (in evaluation mode k is always 1)
                - marginless_logits: A tensor containing the logits for each class in the model, calculated using the
                marginless activations.
                - conv_features: A tensor containing the convolutional features.
                - marginless_max_activations: A tensor containing the max-pooled marginless activations.

        """
        activations, additional_returns = self.prototype_activations(
            x, prototypes_of_wrong_class=prototypes_of_wrong_class
        )
        marginless_activations = additional_returns[0]
        conv_features = additional_returns[1]

        # Set topk_k based on training mode: use predefined value if training, else 1 for evaluation
        if self.training:
            topk_k = self.topk_k
        else:
            topk_k = 1

        # Reshape activations to combine spatial dimensions: (batch_size, num_prototypes, height*width)
        activations = activations.view(activations.shape[0], activations.shape[1], -1)

        # Perform top-k pooling along the combined spatial dimension
        # For topk_k=1, this is equivalent to global max pooling
        topk_activations, _ = torch.topk(activations, topk_k, dim=-1)

        # Calculate the mean of the top-k activations for each channel: (batch_size, num_channels)
        # If topk_k=1, this mean operation does nothing since there's only one value.
        mean_activations = torch.mean(topk_activations, dim=-1)

        marginless_max_activations = nn.functional.max_pool2d(
            marginless_activations,
            kernel_size=(
                marginless_activations.size()[2],
                marginless_activations.size()[3],
            ),
        )
        marginless_max_activations = marginless_max_activations.view(
            -1, self.num_prototypes
        )

        logits = self.last_layer(mean_activations)
        marginless_logits = self.last_layer(marginless_max_activations)
        return logits, [
            mean_activations,
            marginless_logits,
            conv_features,
            marginless_max_activations,
            marginless_activations,
        ]

    def push_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        This method is needed for the pushing operation.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the convolutional features and marginless activations.
        """

        conv_output = self.conv_features(x)
        _, marginless_activations = self.cos_activation(conv_output)
        return conv_output, marginless_activations

    def get_prototype_orthogonalities(
        self, use_part_prototypes: bool = False
    ) -> torch.Tensor:
        """
        Computes the orthogonality loss, encouraging each piece of a prototype to be orthogonal to the others.

        This method is inspired by the paper:
        https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Interpretable_Image_Recognition_by_Constructing_Transparent_Embedding_Space_ICCV_2021_paper.pdf

        Args:
            use_part_prototypes (bool): If True, treats each spatial part of the prototypes as a separate prototype.

        Returns:
            torch.Tensor: A tensor representing the orthogonalities.
        """

        if use_part_prototypes:
            # Normalize prototypes to unit length
            prototype_vector_length = torch.sqrt(
                torch.sum(torch.square(self.prototype_vectors), dim=1, keepdim=True)
                + self.epsilon_val
            )
            normalized_prototypes = self.prototype_vectors / (
                prototype_vector_length + self.epsilon_val
            )
            

            # Calculate total part prototypes per class
            num_part_prototypes_per_class = (
                self.num_prototypes_per_class
                * self.prototype_shape[2]
                * self.prototype_shape[3]
            )

            # Reshape to match class structure
            normalized_prototypes = normalized_prototypes.view(
                self.num_classes,
                self.num_prototypes_per_class,
                self.prototype_shape[1],
                self.prototype_shape[2] * self.prototype_shape[3],
            )

            # Transpose and reshape to treat each spatial part as a separate prototype
            normalized_prototypes = normalized_prototypes.permute(0, 1, 3, 2).reshape(
                self.num_classes, num_part_prototypes_per_class, self.prototype_shape[1]
            )

        else:
            # Normalize prototypes to unit length
            prototype_vectors_reshaped = self.prototype_vectors.view(
                self.num_prototypes, -1
            )
            prototype_vector_length = torch.sqrt(
                torch.sum(torch.square(prototype_vectors_reshaped), dim=1, keepdim=True)
                + self.epsilon_val
            )
            normalized_prototypes = prototype_vectors_reshaped / (
                prototype_vector_length + self.epsilon_val
            )

            # Reshape to match class structure
            normalized_prototypes = normalized_prototypes.view(
                self.num_classes,
                self.num_prototypes_per_class,
                self.prototype_shape[1]
                * self.prototype_shape[2]
                * self.prototype_shape[3],
            )

            

        # Compute orthogonality matrix for each class
        orthogonalities = torch.matmul(
            normalized_prototypes, normalized_prototypes.transpose(1, 2)
        )

        # Identity matrix to enforce orthogonality
        identity_matrix = (
            torch.eye(normalized_prototypes.shape[1], device=orthogonalities.device)
            .unsqueeze(0)
            .repeat(self.num_classes, 1, 1)
        )

        # Subtract identity to focus on orthogonality
        orthogonalities = orthogonalities - identity_matrix

        return orthogonalities

    def identify_prototypes_to_prune(self) -> List[int]:
        """
        Identifies the indices of prototypes that should be pruned.

        This function iterates through the prototypes and checks if the specific weight
        connecting the prototype to its class is zero. It is specifically designed to handle
        the LinearLayerWithoutNegativeConnections where each class has a subset of features
        it connects to.

        Returns:
            List[int]: A list of prototype indices that should be pruned.
        """
        prototypes_to_prune = []

        # Calculate the number of prototypes assigned to each class
        prototypes_per_class = self.num_prototypes // self.num_classes

        if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
            # Custom layer mapping prototypes to a subset of input features for each output class
            for prototype_index in range(self.num_prototypes):
                class_index = self.prototype_class_identity[prototype_index]
                # Calculate the specific index within the 'features_per_output_class' for this prototype
                index_within_class = prototype_index % prototypes_per_class
                # Check if the specific weight connecting the prototype to its class is zero
                if self.last_layer.weight[class_index, index_within_class] == 0.0:
                    prototypes_to_prune.append(prototype_index)
        else:
            # Standard linear layer: each prototype directly maps to a feature index
            weights_to_check = self.last_layer.weight
            for prototype_index in range(self.num_prototypes):
                class_index = self.prototype_class_identity[prototype_index]
                if weights_to_check[class_index, prototype_index] == 0.0:
                    prototypes_to_prune.append(prototype_index)

        return prototypes_to_prune

    def prune_prototypes_by_threshold(self, threshold: float = 1e-3) -> None:
        """
        Prune the weights in the classification layer by setting weights below a specified threshold to zero.

        This method modifies the weights of the last layer of the model in-place. Weights falling below the
        threshold are set to zero, diminishing their influence in the model's decisions. It also identifies
        and prunes prototypes based on these updated weights, thereby refining the model's structure.

        Args:
            threshold (float): The threshold value below which weights will be set to zero. Defaults to 1e-3.
        """
        # Access the weights of the last layer
        weights = self.last_layer.weight.data

        # Set weights below the threshold to zero
        # This step reduces the influence of low-value weights in the model's decision-making process
        weights[weights < threshold] = 0.0

        # Update the weights in the last layer to reflect the pruning
        self.last_layer.weight.data.copy_(weights)

        # Identify prototypes that need to be pruned based on the updated weights
        prototypes_to_prune = self.identify_prototypes_to_prune()

        # Execute the pruning of identified prototypes
        self.prune_prototypes_by_index(prototypes_to_prune)

    def prune_prototypes_by_index(self, prototypes_to_prune: List[int]) -> None:
        """
        Prunes specified prototypes from the PPNet.

        Args:
            prototypes_to_prune (List[int]): A list of indices indicating the prototypes to be removed.
                                             Each index should be in the range [0, current number of prototypes - 1].

        Returns:
            None
        """

        # Validate the provided indices to ensure they are within the valid range
        if any(
            index < 0 or index >= self.num_prototypes for index in prototypes_to_prune
        ):
            raise ValueError("Provided prototype indices are out of valid range!")

        # Calculate the new number of prototypes after pruning
        self.num_prototypes_after_pruning = self.num_prototypes - len(
            prototypes_to_prune
        )

        # Remove the prototype vectors that are no longer needed
        with torch.no_grad():
            # If frequency_weights are being used, set the weights of pruned prototypes to -7
            if self.frequency_weights is not None:
                self.frequency_weights.data[prototypes_to_prune, :] = -7.0

            # Adjust the weights in the last layer depending on its type
            if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
                # For LinearLayerWithoutNegativeConnections, set the connection weights to zero
                # only for the pruned prototypes related to their specific classes
                for class_idx in range(self.last_layer.out_features):
                    # Identify prototypes belonging to the current class
                    indices_for_class = [
                        idx % self.last_layer.features_per_output_class
                        for idx in prototypes_to_prune
                        if self.prototype_class_identity[idx] == class_idx
                    ]
                    self.last_layer.weight.data[class_idx, indices_for_class] = 0.0
            else:
                # For other layer types, set the weights of pruned prototypes to zero
                self.last_layer.weight.data[:, prototypes_to_prune] = 0.0


    def set_last_layer_incorrect_connection(
        self, incorrect_strength: Optional[float] = None
    ) -> None:
        """
        Modifies the last layer weights to have incorrect connections with a specified strength.
        If incorrect_strength is None, initializes the weights for LinearLayerWithoutNegativeConnections
        with correct_class_connection value.

        Args:
        - incorrect_strength (Optional[float]): The strength of the incorrect connections.
                                                If None, initialize without incorrect connections.

        Returns:
            None
        """
        if incorrect_strength is None:
            # Handle LinearLayerWithoutNegativeConnections initialization
            if isinstance(self.last_layer, LinearLayerWithoutNegativeConnections):
                # Initialize all weights to the correct_class_connection value
                self.last_layer.weight.data.fill_(self.correct_class_connection)
            else:
                raise ValueError(
                    "last_layer is not an instance of LinearLayerWithoutNegativeConnections"
                )

        else:
            # Create a one-hot matrix for correct connections
            positive_one_weights_locations = torch.zeros(
                self.num_classes, self.num_prototypes
            )
            positive_one_weights_locations[
                self.prototype_class_identity,
                torch.arange(self.num_prototypes),
            ] = 1

            # Create a matrix for incorrect connections
            negative_one_weights_locations = 1 - positive_one_weights_locations

            # This variable represents the strength of the connection for correct class
            correct_class_connection = self.correct_class_connection

            # This variable represents the strength of the connection for incorrect class
            incorrect_class_connection = incorrect_strength

            # Modify weights to have correct and incorrect connections
            self.last_layer.weight.data.copy_(
                correct_class_connection * positive_one_weights_locations
                + incorrect_class_connection * negative_one_weights_locations
            )

        if self.last_layer.bias is not None:
            # Initialize all biases to bias_last_layer value
            self.last_layer.bias.data.fill_(self.bias_last_layer)

    def _initialize_weights(self) -> None:
        """
        Initializes the weights of the add-on layers of the network and the last layer with incorrect connections.

        Returns:
            None
        """

        for m in self.add_on_layers.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize the last layer with incorrect connections using specified incorrect class connection strength
        self.set_last_layer_incorrect_connection(
            incorrect_strength=self.incorrect_class_connection # sind abgestellt, keine connections 
        )


class VIT_ppnet(L.LightningModule,VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 ema_update_rate,
                 ppnet_cfg,
    ):
        
        L.LightningModule.__init__(self)

        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), ###test!!
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        self.save_hyperparameters()

        self.ppnet_cfg = ppnet_cfg
        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes,
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    #   for p in model.backbone_model.parameters():
    #     p.requires_grad = False
    # for p in model.add_on_layers.parameters():
    #     p.requires_grad = True
    # model.prototype_vectors.requires_grad = True      
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 
        self.ema_update_rate = ema_update_rate

        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length
        


        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
    

        
        self.class_mask = None
        
        del self.head
        #del self.norm
        del self.fc_norm
        del self.head_drop
        

    def forward_features(self, x):
        B = x.shape[0]
        #x = x.permute(0,1,3,2) # test!!
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) #????
        x = self.norm(x)

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = x[:, 0, :]
            x_patch = x[:, 1:, :] 
            z_f = x_patch - x_cls.unsqueeze(1) 
            try:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 32)
            except:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 64) # audioset
        else:
            x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, 8, 32)

        logits,_ = self.ppnet(x)

        return logits
    

    def forward(self, x):
        logits = self.forward_features(x)
        return logits 


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        logits = self(audio)
        targets = targets.long()
        #preds = logits.sigmoid()
        bce_loss = self.loss(logits, targets.float())
        orthogonality_loss = self.calculate_orthogonality_loss()

        self.log('bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('orthogonality_loss', orthogonality_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = bce_loss + orthogonality_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.update()

        return loss

    def validation_step(self, batch, batch_idx):
        pass
    
    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]

        if self.ema: 
            self.ema.apply_shadow()

        self.mask_t_prob = 0.0
        self.mask_f_prob = 0.0 #fix later!

        pred = self(audio)
        if self.class_mask: 
        # if targets.shape == pred.shape:
        #     targets = targets[:, self.class_mask]
            pred = pred[:, self.class_mask]

        targets = targets.long()
        try:
            loss  = self.loss(pred, targets)
        except:
            loss = self.loss(pred, targets.float())
        
        self.test_predictions.append(pred.detach().cpu())
        self.test_targets.append(targets.detach().cpu())

        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        if self.ema: 
            self.ema.restore()
    
    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        pass
    
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            if "encoder_ema.cls_token" not in pre_state_dict: # without mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        new_key = key[len("encoder."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            else: # with mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        continue
                    elif key.startswith("projectors."):
                        continue
                    elif key.startswith("predictors."):
                        continue
                    elif key.startswith("encoder_ema."):
                        # Remove the "encoder_ema." prefix
                        new_key = key[len("encoder_ema."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            for k in ['head.weight', 'head.bias']:
                if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]
            
            info = self.load_state_dict(pretrained_state_dict, strict=False)

            if not self.class_mask:
                for k in ['head.weight', 'head.bias']:
                    if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del pretrained_state_dict[k]

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            info = self.load_state_dict(pretrained_state_dict, strict=False)

            try:
                trunc_normal_(self.head.weight, std=2e-5)
            except:
                print("no head")


    def calculate_orthogonality_loss(self) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        orthogonalities = self.ppnet.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # Normalize the orthogonality loss by the number of elements
        normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()

        return normalized_orthogonality_loss





class VIT_ppnet(L.LightningModule,VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool,
                 norm_layer,
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes,
                 optimizer,
                 scheduler,
                 pretrained_weights_path, 
                 target_length,
                 ema_update_rate,
                 ppnet_cfg,
    ):
        
        L.LightningModule.__init__(self)

        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), ###test!!
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes,
            drop_path_rate=drop_path,
        )
        self.save_hyperparameters()

        self.ppnet_cfg = ppnet_cfg
        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes,
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    #   for p in model.backbone_model.parameters():
    #     p.requires_grad = False
    # for p in model.add_on_layers.parameters():
    #     p.requires_grad = True
    # model.prototype_vectors.requires_grad = True      
        self.img_size = (img_size_x, img_size_y)
        self.global_pool = global_pool

        norm_layer = partial(nn.LayerNorm, eps=eps)
        self.fc_norm = norm_layer(embed_dim)

        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes 
        self.qkv_bias = qkv_bias 
        self.ema_update_rate = ema_update_rate

        self.optimizer = None
        self.optimizer_cfg = optimizer.target
        self.train_batch_size = optimizer.extras.train_batch_size
        self.layer_decay = optimizer.extras.layer_decay
        self.decay_type = optimizer.extras.decay_type
        self.scheduler_cfg = scheduler

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length
        


        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []
    

        
        self.class_mask = None
        
        del self.head
        #del self.norm
        del self.fc_norm
        del self.head_drop
        

    def forward_features(self, x):
        B = x.shape[0]
        #x = x.permute(0,1,3,2) # test!!
        x = self.patch_embed(x) # batch, patch, embed
        x = x + self.pos_embed[:, 1:, :] # strange
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) #????
        x = self.norm(x)

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = x[:, 0, :]
            x_patch = x[:, 1:, :] 
            z_f = x_patch - x_cls.unsqueeze(1) 
            try:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 32)
            except:
                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 64) # audioset
        else:
            x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, 8, 32)

        logits,_ = self.ppnet(x)

        return logits
    

    def forward(self, x):
        logits = self.forward_features(x)
        return logits 


    def training_step(self, batch, batch_idx):
        audio = batch["audio"]
        targets = batch["label"]
        logits = self(audio)
        targets = targets.long()
        #preds = logits.sigmoid()
        bce_loss = self.loss(logits, targets.float())
        orthogonality_loss = self.calculate_orthogonality_loss()

        self.log('bce_loss', bce_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('orthogonality_loss', orthogonality_loss, on_step=True, on_epoch=True, prog_bar=True)

        loss = bce_loss + orthogonality_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        pass
    
    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        pass
    
    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            #self.patch_embed = PatchEmbed_org(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            try:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except:
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]

            pretrained_state_dict = {}

            if "encoder_ema.cls_token" not in pre_state_dict: # without mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        new_key = key[len("encoder."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            else: # with mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        # Skip any key that starts with "decoder."
                        continue
                    elif key.startswith("encoder."):
                        # Remove the "encoder." prefix
                        continue
                    elif key.startswith("projectors."):
                        continue
                    elif key.startswith("predictors."):
                        continue
                    elif key.startswith("encoder_ema."):
                        # Remove the "encoder_ema." prefix
                        new_key = key[len("encoder_ema."):]
                    else:
                        # Use the original key if no prefix
                        new_key = key
                    
                    # Add the modified key-value pair to the new state dict
                    pretrained_state_dict[new_key] = value

            for k in ['head.weight', 'head.bias']:
                if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]
            
            info = self.load_state_dict(pretrained_state_dict, strict=False)

            if not self.class_mask:
                for k in ['head.weight', 'head.bias']:
                    if k in pretrained_state_dict: #and pretrained_state_dict[k].shape != self.state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del pretrained_state_dict[k]

            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            #patch_hw = (img_size[0] // 16, img_size[1] // 16) 
            pos_embed = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) # not trained, overwrite from sincos
            self.pos_embed.data = torch.from_numpy(pos_embed).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds

            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16) # no overlap. stride=img_size=16
           
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches =  num_patches = self.patch_embed.num_patches # audioset
            #num_patches = 512 # assume audioset, 1024//16=64, 128//16=8, 512=64x8
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except:
                pre_state_dict = checkpoint["state_dict"]

            pretrained_state_dict = {}

            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    # Skip any key that starts with "decoder."
                    continue
                elif key.startswith("encoder."):
                    # Remove the "encoder." prefix
                    new_key = key[len("encoder."):]
                else:
                    # Use the original key if no prefix
                    new_key = key
                
                # Add the modified key-value pair to the new state dict
                pretrained_state_dict[new_key] = value

            state_dict = self.state_dict()

            for k in ["head.weight", "head.bias"]:
                if k in pretrained_state_dict and pretrained_state_dict[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del pretrained_state_dict[k]

            info = self.load_state_dict(pretrained_state_dict, strict=False)

            try:
                trunc_normal_(self.head.weight, std=2e-5)
            except:
                print("no head")


    def calculate_orthogonality_loss(self) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        orthogonalities = self.ppnet.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # Normalize the orthogonality loss by the number of elements
        normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()

        return normalized_orthogonality_loss
    



class VIT_ppnet_plain(VisionTransformer):

    def __init__(self, 
                 img_size_x,
                 img_size_y,
                 patch_size,
                 in_chans,
                 embed_dim,
                 global_pool, # Retained as it's set as an attribute, VisionTransformer might use its default or this instance var
                 norm_layer, # Passed to VisionTransformer
                 mlp_ratio,
                 qkv_bias,
                 eps,
                 drop_path,
                 num_heads,
                 depth,
                 num_classes, # Passed to VisionTransformer, though self.head is deleted later
                 pretrained_weights_path, 
                 target_length,
                 ppnet_cfg,
    ):
        
        # L.LightningModule.__init__(self) # Removed

        
        VisionTransformer.__init__(
            self,
            img_size = (img_size_x, img_size_y), ###test!!
            patch_size = patch_size,
            in_chans = in_chans,
            embed_dim = embed_dim,
            depth = depth,
            num_heads = num_heads,
            mlp_ratio = mlp_ratio,
            qkv_bias = qkv_bias,
            norm_layer = partial(nn.LayerNorm, eps=eps),
            num_classes = num_classes, # ViT's head is deleted, but num_classes might be used by ViT's __init__ for other things
            drop_path_rate=drop_path,
            # global_pool is not passed here, ViT will use its default. self.global_pool is set below.
        )
        # self.save_hyperparameters() # Removed

        self.ppnet_cfg = ppnet_cfg
        self.ppnet = PPNet(
            num_prototypes=ppnet_cfg.num_prototypes,
            channels_prototypes=ppnet_cfg.channels_prototypes,
            h_prototypes=ppnet_cfg.h_prototypes,
            w_prototypes=ppnet_cfg.w_prototypes,
            num_classes=ppnet_cfg.num_classes, # This num_classes is for PPNet
            topk_k=ppnet_cfg.topk_k,
            margin=ppnet_cfg.margin,
            init_weights=ppnet_cfg.init_weights,
            add_on_layers_type=ppnet_cfg.add_on_layers_type,
            incorrect_class_connection=ppnet_cfg.incorrect_class_connection,
            correct_class_connection=ppnet_cfg.correct_class_connection,
            bias_last_layer=ppnet_cfg.bias_last_layer,
            non_negative_last_layer=ppnet_cfg.non_negative_last_layer,
            embedded_spectrogram_height=ppnet_cfg.embedded_spectrogram_height,
        )

    #   for p in model.backbone_model.parameters():
    #     p.requires_grad = False
    # for p in model.add_on_layers.parameters():
    #     p.requires_grad = True
    # model.prototype_vectors.requires_grad = True      
        self.img_size = (img_size_x, img_size_y) # Already set by VisionTransformer.__init__ if img_size is passed
        self.global_pool = global_pool # Sets attribute, ViT uses its own default if not passed in its __init__

        # norm_layer is passed to VisionTransformer, self.fc_norm is deleted anyway
        # self.fc_norm = norm_layer(embed_dim) # Removed as self.fc_norm is deleted


        # These are mostly set by VisionTransformer.__init__ or are specific to this class
        self.embed_dim = embed_dim 
        self.num_heads = num_heads
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.num_classes = num_classes # This is ViT's num_classes, PPNet has its own
        self.qkv_bias = qkv_bias 
        # self.ema_update_rate = ema_update_rate # Removed

        # Optimizer and scheduler related attributes removed
        # self.optimizer = None
        # self.optimizer_cfg = optimizer.target
        # self.train_batch_size = optimizer.extras.train_batch_size
        # self.layer_decay = optimizer.extras.layer_decay
        # self.decay_type = optimizer.extras.decay_type
        # self.scheduler_cfg = scheduler

        self.pretrained_weights_path = pretrained_weights_path
        self.target_length = target_length
        
        # Prediction/target lists for Lightning validation/test loops removed
        # self.val_predictions = []
        # self.val_targets = []
        # self.test_predictions = []
        # self.test_targets = []
    
        self.class_mask = None # Retained, used in load_pretrained_weights
        
        # Deletions are part of the model architecture, kept
        del self.head
        #del self.norm # self.norm is used in forward_features
        if hasattr(self, 'fc_norm'): # fc_norm might not exist if global_pool is not 'avg' or 'token' in ViT
            del self.fc_norm
        del self.head_drop
        

    def forward_features(self, x):
        B = x.shape[0]
        #x = x.permute(0,1,3,2) # test!!
        x = self.patch_embed(x) # batch, patch, embed
        
        # Original ViT adds pos_embed to concatenated cls_token and patch_embed
        # x = self.pos_drop(self._pos_embed(x)) 
        # This implementation is slightly different:
        x = x + self.pos_embed[:, 1:, :] # Add pos_embed to patch tokens
        cls_token = self.cls_token + self.pos_embed[:, :1, :] # Add pos_embed to cls_token
        
        cls_tokens = cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)        

        for blk in self.blocks:
            x = blk(x)
            #x = torch.nan_to_num(x, nan=0.0) #????
        x = self.norm(x) # self.norm is from VisionTransformer

        if self.ppnet_cfg.focal_similarity == True:
            x_cls = x[:, 0, :]
            x_patch = x[:, 1:, :] 
            z_f = x_patch - x_cls.unsqueeze(1) 
            try:
                # Determine target shape based on target_length or other config if available
                # Assuming 8 is fixed height for prototype (e.g. 128 / 16 = 8)
                # Width depends on target_length (e.g. 512 / 16 = 32, 1024 / 16 = 64)
                # This should ideally be derived from self.patch_embed.grid_size or similar
                # For now, keeping the try-except based on common values
                if self.target_length == 512: # Example, adjust as needed
                    width_patches = 32
                elif self.target_length == 1024: # Example, adjust as needed
                    width_patches = 64
                else: # Fallback or error, this part might need more robust logic
                    # Assuming default from original code if not 512 or 1024
                    # This logic is fragile and depends on specific target_lengths
                    # A more robust way would be to calculate H_patch, W_patch from img_size and patch_size
                    # e.g., H_patch = self.img_size[1] // self.patch_embed.patch_size[1]
                    #       W_patch = self.img_size[0] // self.patch_embed.patch_size[0]
                    # For now, replicating the existing try-except logic's implied behavior
                    if x_patch.shape[1] == 256: # 8 * 32
                         width_patches = 32
                    elif x_patch.shape[1] == 512: # 8 * 64
                         width_patches = 64
                    else:
                        # Fallback, or raise error if shape is unexpected
                        # This was the original logic's implicit assumption
                        # For a robust solution, calculate embedded_h, embedded_w
                        # embedded_h = self.img_size[1] // self.patch_embed.patch_size[1] # e.g. 128 // 16 = 8
                        # embedded_w = self.img_size[0] // self.patch_embed.patch_size[0] # e.g. 512 // 16 = 32 or 1024 // 16 = 64
                        # x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, embedded_h, embedded_w)
                        # For now, keeping the try-except structure as it was
                        raise ValueError(f"Unexpected number of patches for focal similarity: {x_patch.shape[1]}")

                x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, width_patches)

            except RuntimeError as e: # More specific error handling if possible
                 # This fallback is based on the original code's structure
                 # It implies that if the first reshape fails, it tries another common one.
                 # This is not ideal. The shape should be deterministic.
                if "32" in str(e): # If error was about 32, try 64
                    x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 64) # audioset
                elif "64" in str(e): # If error was about 64, try 32
                    x = z_f.permute(0, 2, 1).reshape(B, self.embed_dim, 8, 32)
                else:
                    raise e # Re-raise if it's not the expected sizing issue
        else:
            # Similar to above, the reshape dimensions should be robustly determined
            # embedded_h = self.img_size[1] // self.patch_embed.patch_size[1]
            # embedded_w = self.img_size[0] // self.patch_embed.patch_size[0]
            # x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, embedded_h, embedded_w)
            # Replicating original hardcoded values for now
            if x.shape[1]-1 == 256: # num_patches = 8 * 32
                x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, 8, 32)
            elif x.shape[1]-1 == 512: # num_patches = 8 * 64
                x = x[:,1:,:].permute(0,2,1).reshape(B, self.embed_dim, 8, 64) # audioset
            else:
                # Fallback or error
                raise ValueError(f"Unexpected number of patches for non-focal similarity: {x.shape[1]-1}")


        logits,_ = self.ppnet(x)

        return logits
    

    def forward(self, x):
        logits = self.forward_features(x)
        return logits 

    # training_step, validation_step, on_validation_epoch_end, test_step, on_test_epoch_end, configure_optimizers removed
    # as they are PyTorch Lightning specific.
    # The user will need to implement their own training loop.

    def load_pretrained_weights(self, pretrained_weights_path, dataset_name): 
        img_size = (self.target_length, 128)
        #img_size = (128, self.target_length) # should be correcter, but not pretrained this way

        if self.target_length == 512: #esc50, hsn, 5 seconds
            #num_patches = 512 # audioset
            if "xc" in self.pretrained_weights_path or "XCL" in self.pretrained_weights_path:
                num_patches = 256 # birdset
            else:
                num_patches = 512 # audioset

            # Re-initialize patch_embed and pos_embed if they depend on num_patches that might change
            # This was already in the original code, so keeping it.
            self.patch_embed = PatchEmbed(img_size, 16, 1, self.embed_dim)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False) #to load pretrained pos embed
            
            try:
                # Try loading assuming 'model' key, common in many checkpoints
                pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["model"]
            except KeyError:
                # Fallback for checkpoints that might store it directly or under 'state_dict'
                try:
                    pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")["state_dict"]
                except KeyError:
                    pre_state_dict = torch.load(pretrained_weights_path, map_location="cpu")


            pretrained_state_dict_processed = {}

            if "encoder_ema.cls_token" not in pre_state_dict: # without mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder."):
                        continue
                    elif key.startswith("encoder."):
                        new_key = key[len("encoder."):]
                    else:
                        new_key = key
                    pretrained_state_dict_processed[new_key] = value
            else: # with mim refiner
                for key, value in pre_state_dict.items():
                    if key.startswith("decoder.") or \
                       key.startswith("encoder.") or \
                       key.startswith("projectors.") or \
                       key.startswith("predictors."):
                        continue
                    elif key.startswith("encoder_ema."):
                        new_key = key[len("encoder_ema."):]
                    else:
                        new_key = key
                    pretrained_state_dict_processed[new_key] = value
            
            current_model_state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']: # These keys might not exist if self.head was deleted
                if k in pretrained_state_dict_processed:
                    # Check if the key also exists in the current model (it shouldn't if head is deleted)
                    # Or if shapes mismatch (though if head is deleted, this check is moot for head keys)
                    if k not in current_model_state_dict or \
                       pretrained_state_dict_processed[k].shape != current_model_state_dict.get(k, None).shape:
                        print(f"Removing key {k} from pretrained checkpoint (mismatch or not in current model)")
                        del pretrained_state_dict_processed[k]
            
            info = self.load_state_dict(pretrained_state_dict_processed, strict=False)
            print(f"Pretrained weights loading info: {info}")


            # This part seems to be re-initializing pos_embed based on sincos,
            # potentially overwriting loaded weights if 'pos_embed' was in the checkpoint.
            # This is kept as per original logic.
            patch_hw = (img_size[1] // 16, img_size[0] // 16) # 16=patchsize
            pos_embed_sincos = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True) 
            self.pos_embed.data = torch.from_numpy(pos_embed_sincos).float().unsqueeze(0) 

        elif self.target_length == 1024: #audioset, 10 seconds
            # Re-initialize patch_embed and pos_embed
            self.patch_embed = PatchEmbed_new(img_size=img_size, patch_size=(16,16), in_chans=1, embed_dim=self.embed_dim, stride=16)
            
            # num_patches logic seems to have a duplicate assignment, corrected
            if "xc" in self.pretrained_weights_path:
                num_patches = 256 # birdset # does not work right now 
            else:
                num_patches = self.patch_embed.num_patches # audioset
            
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding

            checkpoint = torch.load(pretrained_weights_path, map_location="cpu")
            try:
                pre_state_dict = checkpoint["model"]
            except KeyError:
                try:
                    pre_state_dict = checkpoint["state_dict"]
                except KeyError:
                    pre_state_dict = checkpoint


            pretrained_state_dict_processed = {}
            for key, value in pre_state_dict.items():
                if key.startswith("decoder."):
                    continue
                elif key.startswith("encoder."):
                    new_key = key[len("encoder."):]
                else:
                    new_key = key
                pretrained_state_dict_processed[new_key] = value

            current_model_state_dict = self.state_dict()
            for k in ["head.weight", "head.bias"]: # These keys might not exist if self.head was deleted
                if k in pretrained_state_dict_processed:
                    if k not in current_model_state_dict or \
                       pretrained_state_dict_processed[k].shape != current_model_state_dict.get(k, None).shape:
                        print(f"Removing key {k} from pretrained checkpoint (mismatch or not in current model)")
                        del pretrained_state_dict_processed[k]

            info = self.load_state_dict(pretrained_state_dict_processed, strict=False)
            print(f"Pretrained weights loading info: {info}")
            
            # For 1024 target_length, pos_embed is also re-initialized with sincos after loading attempt
            # This is consistent with the 512 case.
            patch_hw = (img_size[1] // 16, img_size[0] // 16)
            pos_embed_sincos = get_2d_sincos_pos_embed_flexible(self.pos_embed.size(-1), patch_hw, cls_token=True)
            self.pos_embed.data = torch.from_numpy(pos_embed_sincos).float().unsqueeze(0)

            # The original code had a try-except for trunc_normal_ on self.head.weight.
            # Since self.head is deleted, this is no longer applicable.
            # try:
            #     trunc_normal_(self.head.weight, std=2e-5)
            # except AttributeError: # More specific exception
            #     print("No head attribute to initialize with trunc_normal_ (as expected).")
            # except Exception as e:
            #     print(f"Error during trunc_normal_ on head (unexpected): {e}")


    def calculate_orthogonality_loss(self) -> torch.Tensor:
        """
        Calculate the normalized orthogonality loss.

        Returns:
            torch.Tensor: The normalized orthogonality loss.
        """
        orthogonalities = self.ppnet.get_prototype_orthogonalities()
        orthogonality_loss = torch.norm(orthogonalities)

        # Normalize the orthogonality loss by the number of elements
        if orthogonalities.numel() > 0:
            normalized_orthogonality_loss = orthogonality_loss / orthogonalities.numel()
        else:
            normalized_orthogonality_loss = torch.tensor(0.0, device=orthogonality_loss.device)


        return normalized_orthogonality_loss


# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import numpy as np

import torch

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32) # grid size[0] = 8
    grid_w = np.arange(grid_size[1], dtype=np.float32) # grid size[1] = 32
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0) # 2,8,32

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]]) # 2,1,8.32
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed # 267 (+cls) x 1024 (feature dim)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def interpolate_pos_embed_img2audio(model, checkpoint_model, orig_size, new_size):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        #orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        #new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def interpolate_pos_embed_audio(model, checkpoint_model, orig_size, new_size):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size[0], orig_size[1], new_size[0], new_size[1]))
            #extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            cls_token = pos_embed_checkpoint[:,0,:].unsqueeze(1)
            pos_tokens = pos_embed_checkpoint[:,1:,:] # remove 
            pos_tokens = pos_tokens.reshape(-1, orig_size[0], orig_size[1], embedding_size) #.permute(0, 3, 1, 2)
            #pos_tokens = torch.nn.functional.interpolate(
            #    pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)
            
            #pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            pos_tokens = pos_tokens[:,:,:new_size[1],:] # assume only time diff
            pos_tokens = pos_tokens.flatten(1, 2)
            new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


def interpolate_patch_embed_audio(model, checkpoint_model, orig_channel, new_channel=1, kernel_size=(16,16), stride=(16,16), padding=(0,0)):
    if orig_channel != new_channel:
        if 'patch_embed.proj.weight' in checkpoint_model:
            # aggregate 3 channels in rgb ckpt to 1 channel for audio
            new_proj_weight = torch.nn.Parameter(torch.sum(checkpoint_model['patch_embed.proj.weight'], dim=1).unsqueeze(1))
            checkpoint_model['patch_embed.proj.weight'] = new_proj_weight

import torch
import torch.nn as nn
from timm.models.layers import to_2tuple

class PatchEmbed_new(nn.Module): # OVERLAPPED PATCHES
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        #self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        #self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        # todo: don't be lazy..
        return self.proj(torch.randn(1,1,img_size[0],img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x