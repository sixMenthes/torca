import torch
class EMA:
    def __init__(self, model, decay=0.9998):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.shadow_buffers = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow_params[name] = param.detach().clone()
                    self.shadow_params[name].requires_grad = False
            for name, buffer in model.named_buffers():
                self.shadow_buffers[name] = buffer.detach().clone()

        # We'll store the base model's original weights here (for restore)
        self.backup_params = {}
        self.backup_buffers = {}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = self.shadow_params[name].to(param.device)
                self.shadow_params[name].copy_(
                    self.decay * self.shadow_params[name] + (1.0 - self.decay) * param.data
                )
        for name, buffer in self.model.named_buffers():
            self.shadow_buffers[name] = buffer.detach().clone().to(buffer.device)

    @torch.no_grad()
    def apply_shadow(self):
        """
        1) Backup the base model's current params/buffers
        2) Load the shadow (EMA) weights into the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # backup base weights
                self.backup_params[name] = param.detach().clone()
                # load shadow
                param.copy_(self.shadow_params[name].to(param.device))

        for name, buffer in self.model.named_buffers():
            # backup base buffers
            self.backup_buffers[name] = buffer.detach().clone()
            # load shadow
            buffer.copy_(self.shadow_buffers[name].to(buffer.device))

    @torch.no_grad()
    def restore(self):
        """
        Restore the base model weights from `backup_...`
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup_params:
                param.copy_(self.backup_params[name].to(param.device))
        for name, buffer in self.model.named_buffers():
            if name in self.backup_buffers:
                buffer.copy_(self.backup_buffers[name].to(buffer.device))