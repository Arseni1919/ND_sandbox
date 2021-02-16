from IMPORTS import *
from alg_dataset import alg_dataset


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),

        )
        self.net.double()
        self.counter = 0

    def forward(self, x):
        x = self.net(x)
        return x

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.view(-1)
        loss = F.mse_loss(y_hat, y)  # F.mse_loss(y_hat, y.float())

        self.log('train loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    # def train_dataloader(self):
    #     return DataLoader(alg_dataset, batch_size=64)


alg_lit_module = ALGLightningModule()


