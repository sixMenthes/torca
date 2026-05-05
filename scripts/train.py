import time
from torch import nn
from torch.optim import AdamW
# Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (ICLR 2019) — https://arxiv.org/abs/1711.05101

from transformers import get_cosine_schedule_with_warmup
# Loshchilov & Hutter, "SGDR: Stochastic Gradient Descent with Warm Restarts" (ICLR 2017) — https://arxiv.org/abs/1608.03983
from torca import Torca
from dataset import LocalDataset, collate_fn
from torch.utils.data import DataLoader
from cfg import TorcaConfig
import polars as pl

def make_model(cfg):
    model = Torca(cfg)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0

def run_epoch(data_iter,
    model,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=TrainState(),
    ):

    for i, batch in enumerate(data_iter):
        total_loss = 0
        n_accum = 0
        model.train()
        loss = model.forward(batch.src, batch.tgt, batch.src_mask)
        if mode == "train":
            (loss/accum_iter).backward()
            train_state.step += 1
            train_state.samples += batch.src.size(0)
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                train_state.accum_step += 1
                n_accum += 1
            scheduler.step() 
        total_loss += loss
        if i % 40 == 1 and (mode == "train"):
            lr = optimizer.param_groups[0]["lr"]
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Learning Rate: %6.1e"
                    )
                    % (i, n_accum, loss, lr)
            )
            del loss
        return total_loss

def train(train_data, val_data):
    cfg = TorcaConfig()
    model = make_model(cfg)
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)
    batch_size = cfg["batch_size"]
    for epoch in range(cfg["num_epochs"]):
        model.train()
        train_loss = run_epoch(
            train_data, 
            model, 
            optimizer,
            scheduler="hi",
        )
        val_loss = run_epoch(
            val_data,
            model,
            optimizer,
            scheduler="hi"
        )


    



        
