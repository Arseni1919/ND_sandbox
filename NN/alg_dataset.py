from IMPORTS import *
from y_func import y_func


class ALGDataset(Dataset):
    def __init__(self, func):
        self.func = func
        self.buffer = deque(maxlen=REPLAY_SIZE)
        for i in range(REPLAY_SIZE):
            x_num = random.uniform(0, SCALE)
            y_num = random.uniform(0, SCALE)
            z_num = self.func(x_num, y_num)
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


class STOCKSDataset(Dataset):
    """
    Each day is 390 entries in the dataframe
    x = stocks_in_data
    y = 'SPY'
    """
    def __init__(self):
        self.data = pd.read_csv('../data/real_data_2days_sample.csv')
        list_of_graphs = []
        for k in self.data.keys():
            if 'Volume' not in k:
                list_of_graphs.append(k)
        self.stocks_in_data = list_of_graphs[16:]
        self.max_SPY = self.data['SPY'].max()
        self.data_len = len(self.data) - WINDOW_TO_LOOK_BACK + 1

    def __len__(self):
        return self.data_len

    def __getitem__(self, indx):
        data_to_return = self.data[indx:indx+WINDOW_TO_LOOK_BACK]
        data_to_return = data_to_return[self.stocks_in_data]
        data_to_return = data_to_return.to_numpy()
        min_max_scaler = preprocessing.MinMaxScaler()
        data_to_return = min_max_scaler.fit_transform(data_to_return)
        obs_x = torch.Tensor(data_to_return).double()
        obs_x = torch.reshape(obs_x, (-1,))
        obs_y = self.data['SPY'][indx + WINDOW_TO_LOOK_BACK - 1]
        obs_y = obs_y / self.max_SPY
        obs = (obs_x, obs_y)
        return obs



