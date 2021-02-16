from main_simple_nn import *

# data = pd.read_csv('../data/real_data_2days_sample.csv')
# print(len(data))

# b = torch.tensor([[0, 1], [2, 3]])
# print(torch.reshape(b, (-1,)))

from alg_dataset import *

sample_dataset = STOCKSDataset()
alg_data_module = ALGDataModule(sample_dataset)
a = sample_dataset.data['SPY'][779:780]
for i in alg_data_module.train_dataloader():
    print(i)