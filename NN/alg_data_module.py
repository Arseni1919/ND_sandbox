from IMPORTS import *


class ALGDataModule(pl.LightningDataModule):

    def __init__(self, alg_dataset):
        super().__init__()
        self.alg_dataset = alg_dataset

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.alg_dataset, batch_size=BATCH_SIZE, num_workers=4)
        # return DataLoader(self.alg_dataset, batch_size=BATCH_SIZE, num_workers=8)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass



