from IMPORTS import *
from alg_dataset import alg_dataset


class ALGDataModule(pl.LightningDataModule):

    def __init__(self, alg_dataset):
        super().__init__()
        self.alg_dataset=alg_dataset

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(self.alg_dataset, batch_size=64)

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass


data_module = ALGDataModule(alg_dataset)

