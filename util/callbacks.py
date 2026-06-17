from lightning.pytorch.callbacks import Callback

class TestAfterEpoch(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.test(model=pl_module, datamodule=trainer.datamodule)