from IMPORTS import *

class ALGLightningModule(pl.LightningModule):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('current total reward', self.total_reward)
        self.log('train loss', loss)
        # self.log('epsilon', epsilon)
        # if batch_idx % 1000 == 0:
        #     print(epsilon)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=LR)