from IMPORTS import *


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
        # self.net = RNN(input_size=RNN_INPUT_SIZE, hidden_size=RNN_HIDDEN_SIZE, output_size=RNN_OUTPUT_SIZE)  GRU
        # self.rnn = nn.RNN(input_size=RNN_INPUT_SIZE,
        #                   hidden_size=RNN_HIDDEN_SIZE,
        #                   num_layers=RNN_NUM_LAYERS,
        #                   batch_first=True)
        self.rnn = nn.GRU(input_size=RNN_INPUT_SIZE,
                          hidden_size=RNN_HIDDEN_SIZE,
                          num_layers=RNN_NUM_LAYERS,
                          batch_first=True)
        # x -> (batch_size, seq_len, input_size)
        self.fc = nn.Linear(RNN_HIDDEN_SIZE, RNN_OUTPUT_SIZE)
        self.rnn.double()
        self.fc.double()
        self.counter = 0

    def forward(self, x):
        h0 = torch.zeros((1 * RNN_NUM_LAYERS, x.size(0), RNN_HIDDEN_SIZE))
        h0 = torch.Tensor(h0).double()

        output, hidden = self.rnn(x, h0)
        # out -> (batch_size, seq_len, hidden_size)
        # out ()
        output = output[:, -1, :]
        output = self.fc(output)
        return output

    def training_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        y_hat = self(x_batch)
        y_real = y_batch[-1]
        loss = F.mse_loss(y_hat, y_real)  # F.mse_loss(y_hat, y.float())

        self.log('train loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    # def train_dataloader(self):
    #     return DataLoader(alg_dataset, batch_size=64)


alg_lit_module = ALGLightningModule()


