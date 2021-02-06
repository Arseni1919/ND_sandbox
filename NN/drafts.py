from main_simple_nn import *

model = ALGLightningModule()
model.load_state_dict(torch.load("example.ckpt"))
compare_graphs(model, y_func)

# b = torch.tensor([[0, 1], [2, 3]])
# print(torch.reshape(b, (-1,)))