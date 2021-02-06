from IMPORTS import *
from alg_dataset import alg_dataset


class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            #             nn.Linear(256, 256),
            #             nn.ReLU(),
            #             nn.Linear(256, 256),
            #             nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 1),
            #             nn.ReLU(),
        )
        #         self.l1 = torch.nn.Linear(2, 100)
        #         self.l2 = torch.nn.Linear(100, 1)
        #         self.l3 = torch.nn.Linear(100, 1)
        self.counter = 0

    def forward(self, x):
        #         return self.l2(torch.relu(self.l1(x)))
        #         x = F.log_softmax(self.net(x), dim=1)
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):
        print(f'\r{self.counter}', end='')
        self.counter += 1
        x, y = batch
        #         print(f'x size: {x.size()}')
        #         print(f'y reshape: {y.reshape(-1).size()}')
        #         print(f'X: {x.float().dtype}')
        #         print(f'Y: {y.float().dtype}')

        y_hat = self(x.float())
        loss = F.mse_loss(y_hat, y.float())
        #         loss = F.cross_entropy(y_hat, y)

        self.log('train loss', loss)
        # if batch_idx % 1000 == 0:
        #     print(epsilon)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def train_dataloader(self):
        return DataLoader(alg_dataset, batch_size=64)


alg_lit_module = ALGLightningModule()
