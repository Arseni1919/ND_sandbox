from IMPORTS import *


class ALGCallback(Callback):

    def on_init_start(self, trainer):
        print('--- Starting to init trainer! ---')

    def on_init_end(self, trainer):
        print('--- trainer is init now ---')

    def on_train_end(self, trainer, pl_module):
        # print('--- training ends ---')
        torch.save(pl_module.state_dict(), "example.ckpt")

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        """Called when the train epoch ends."""
        pass

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        # print('--- print batch ---')
        # for i in batch:
        #     print(i)
        pass