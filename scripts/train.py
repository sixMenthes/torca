import torch
import os
from torch.optim import AdamW
import torch.nn.functional as F
# Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019) — https://arxiv.org/abs/1711.05101

from transformers import get_cosine_schedule_with_warmup
# Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017) — https://arxiv.org/abs/1608.03983
from torca import Torca
from dataset import LocalDataset, collate_fn
from torch.utils.data import DataLoader
from cfg import TorcaConfig
import polars as pl
from sklearn.metrics import average_precision_score
import numpy as np
from utils import get_class_coefs

def make_model(cfg, class_weights):
    model = Torca(cfg, class_weights)
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model

def calc_MAP(y, y_pred):
    n_classes = TorcaConfig.num_classes
    average_precision = {}
    running_total = 0
    for i in range(n_classes):
        score = average_precision_score(y[:, i], y_pred[:, i])
        # we are iterating along the dimension num_classes of the output self.classif_head with size (B, num_classes), and for each class we're computing the AP
        average_precision[i] = score
        running_total += score
    average_precision["mean"] = running_total / n_classes
    return average_precision


class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0

def run_epoch(data_iter,
    model,
    optimizer,
    scheduler,
    class_loss_weight,
    device,
    accum_iter=1,
    train_state=TrainState(),
    mode = "train"
    ):

    n_accum = 0
    total_summed_loss = 0
    total_mask_loss = 0
    total_classification_loss = 0
    output_per_epoch = {
        "summed_loss": [],
        "mask_loss": [],
        "classification_loss": [],
        "probs": [],
        "labels": []
    }

    if mode == "train":
        model.train()

    for i, batch in enumerate(data_iter):
        padded, mask, labels = (t.to(device) for t in batch)
        mask_loss, clas_loss, clas_logits = model.forward(padded, labels, mask) 
        # in order given by collate_fn: padded sw, mask, labels
        # forward expects: padded sw, labels, mask
        
        summed_loss = mask_loss + class_loss_weight * clas_loss
        total_summed_loss += summed_loss.item()
        total_mask_loss += mask_loss.item()
        total_classification_loss += clas_loss.item()

        if mode == "train":
            (summed_loss/accum_iter).backward()
            train_state.step += 1
            train_state.samples += batch[0].size(0)
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                train_state.accum_step += 1
                n_accum += 1
                scheduler.step() 
            if i % 40 == 1:
                lr = optimizer.param_groups[0]["lr"]
                print(
                    (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Learning Rate: %6.1e"
                    )
                    % (i, n_accum, summed_loss.item(), lr)
                )
        
        if mode == "eval":
            output_per_epoch["probs"].append(F.softmax(clas_logits, dim=-1).detach().cpu())
            output_per_epoch["labels"].append(batch[2].detach().cpu())
        
    output_per_epoch["summed_loss"] = total_summed_loss / len(data_iter)
    output_per_epoch["mask_loss"] = total_mask_loss / len(data_iter)
    output_per_epoch["classification_loss"] = total_classification_loss / len(data_iter)

    if mode == "eval":
        output_per_epoch["probs"] = torch.cat(output_per_epoch["probs"]).numpy()
        output_per_epoch["labels"] = F.one_hot(torch.cat(output_per_epoch["labels"]), num_classes=TorcaConfig.num_classes).numpy()

    return output_per_epoch

def train():
    train_df = pl.read_parquet("../ds/DCLDE_train_manifest.parquet").filter(pl.col("Split") == "train")
    train_dataset = LocalDataset(train_df)
    class_weights = torch.from_numpy(get_class_coefs(train_dataset)).float()
    class_maps = train_dataset.label_map
    val_dataset = LocalDataset(pl.read_parquet("../ds/DCLDE_train_manifest.parquet").filter(pl.col("Split") == "val"), class_maps)
    train_loader = DataLoader(train_dataset, 32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, 5, shuffle=False, collate_fn=collate_fn)
    cfg = TorcaConfig()
    model = make_model(cfg, class_weights)
    class_weights = torch.from_numpy(get_class_coefs(train_dataset)).float()
    print("class_weights:", class_weights.shape, class_weights)
    print("label_map:", train_dataset.label_map)
    print("num_classes (cfg):", cfg.num_classes)

    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    class_loss_weight = cfg.class_loss_weight
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler_warmup = int(0.05 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps= scheduler_warmup, num_training_steps=total_steps)
    rows = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)


    for epoch in range(cfg.num_epochs):
        train_output = run_epoch(
            train_loader, 
            model, 
            optimizer,
            scheduler=scheduler,
            class_loss_weight=class_loss_weight,
            device=device,
            mode="train"
        )
        with torch.no_grad():
            model.eval()
            val_output = run_epoch(
                val_loader,
                model,
                optimizer,
                scheduler=scheduler,
                class_loss_weight=class_loss_weight,
                device=device,
                mode="eval"
            )
        map_res = calc_MAP(val_output["labels"], val_output["probs"])
        

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_total_loss": float(np.mean(train_output["summed_loss"])),
            "train_mask_loss": float(np.mean(train_output["mask_loss"])),
            "train_classif_loss": float(np.mean(train_output["classification_loss"])),
            "val_total_loss": float(np.mean(val_output["summed_loss"])),
            "val_mask_loss": float(np.mean(val_output["mask_loss"])),
            "val_classif_loss": float(np.mean(val_output["classification_loss"])),
            "val_mean_precision": map_res["mean"],
            **{f"val_ap_{i}": map_res[i] for i in range(cfg.num_classes)},
            "label_map": [k for k, _ in class_maps.items()]
        }

        rows.append(row)
        os.makedirs("./runs", exist_ok=True)
        pl.DataFrame(rows).write_parquet("./runs/train_logs.parquet")
    
if __name__ == "__main__":
    train()
