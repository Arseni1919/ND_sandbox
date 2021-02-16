from main_help_RNN import *


class ALGDataset(Dataset):
    def __init__(self, func):
        self.func = func
        self.buffer = deque(maxlen=REPLAY_SIZE)
        for i in range(REPLAY_SIZE):
            x_num, y_num, z_num = self.func(i)
            obs_xy = torch.Tensor([x_num, y_num]).double()
            obs = (obs_xy, z_num)
            self.buffer.append(obs)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, indx):
        z_num = self.buffer[indx]
        return z_num

    def append(self, new_z):
        self.buffer.append(new_z)


alg_dataset = ALGDataset(y_func)
