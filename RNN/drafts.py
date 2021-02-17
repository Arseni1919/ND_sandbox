from alg_data_module import *

check_model = True
# check_model = False

# if check_model:
#     from main_RNN import *
#     model = ALGLightningModule()
#     model.load_state_dict(torch.load("example.ckpt"))
#     compare_graphs(model, y_func)
# else:
#     x, y, z = [], [], []
#     for i in range(len(alg_dataset)):
#         obs, z_val = alg_dataset[i]
#         obs = obs.numpy()
#         x_val, y_val = obs
#         x.append(x_val)
#         y.append(y_val)
#         z.append(z_val)
#
#     fig = go.Figure()  # showscale=False, colorbar_x=-0.07,
#     x_axis = list(range(len(x)))
#     fig.add_trace(go.Scatter(x=x_axis, y=x, name='x', opacity=0.1))
#     fig.add_trace(go.Scatter(x=x_axis, y=y, name='y', opacity=0.1))
#     fig.add_trace(go.Scatter(x=x_axis, y=z, name='z'))
#     fig.show()

for i in alg_data_module.train_dataloader():
    print(i)
    print(len(i))
    break


# a = np.zeros(REPLAY_SIZE)
# print(a)
# a[0] = 0.002
# print(a)

# rnn = nn.RNN(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# output, hn = rnn(input, h0)
# print(output)
# print(hn)






