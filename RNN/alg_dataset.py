from IMPORTS import *


class ALGDataset(Dataset):
    def __init__(self, func):
        self.func = func
        # self.buffer = deque(maxlen=REPLAY_SIZE)
        self.buffer_x = np.zeros((REPLAY_SIZE, RNN_INPUT_SIZE))
        self.buffer_y = np.zeros(REPLAY_SIZE)
        for i in range(REPLAY_SIZE):
            x_num, y_num, z_num = self.func(i)
            self.buffer_x[i][0] = x_num
            self.buffer_x[i][1] = y_num
            self.buffer_y[i] = z_num
            # obs_xy = torch.Tensor([x_num, y_num]).double()
            # obs = (obs_xy, z_num)
            # self.buffer.append(obs)

    def __len__(self):
        return REPLAY_SIZE - RNN_SEQ_LEN

    def __getitem__(self, indx):
        xy = self.buffer_x[indx:indx+RNN_SEQ_LEN]
        obs_xy = torch.Tensor(xy).double()
        obs_y = self.buffer_y[indx+RNN_SEQ_LEN]
        return obs_xy, obs_y



