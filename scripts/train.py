import time
from transformers import get_cosine_schedule_with_warmup

class TrainState:
    step: int = 0
    accum_step: int = 0
    samples: int = 0

def run_epoch(data_iter,
    model,
    loss_compute,
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



        
