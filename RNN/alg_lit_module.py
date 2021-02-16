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
        combined = torch.cat((input, hidden), 0)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        # output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

# n_hidden = 128
# rnn = RNN(n_letters, n_hidden, n_categories)


class ALGLightningModule(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = RNN(input_size=RNN_INPUT_SIZE, hidden_size=RNN_HIDDEN_SIZE, output_size=RNN_OUTPUT_SIZE)
        self.net.double()
        self.counter = 0

    def forward(self, input_item, hidden):
        output, hidden = self.net(input_item, hidden)
        return output, hidden

    def training_step(self, batch, batch_idx):
        hidden = torch.zeros(RNN_HIDDEN_SIZE)
        y, output = 0, 0
        # output, next_hidden = rnn(input, hidden)
        input_batch, y_batch = batch
        for input_item_indx, input_item in enumerate(input_batch):
            output, hidden = self(input_item, hidden)
            y = y_batch[input_item_indx]
        y_hat = output.view(-1)
        loss = F.mse_loss(y_hat, y)  # F.mse_loss(y_hat, y.float())

        self.log('train loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    # def train_dataloader(self):
    #     return DataLoader(alg_dataset, batch_size=64)


alg_lit_module = ALGLightningModule()


