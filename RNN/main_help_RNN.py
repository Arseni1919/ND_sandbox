from IMPORTS import *
from alg_dataset import ALGDataset

def y_func(i):
    """
    :param a: between [0, 1]
    :param b: between [0, 1]
    :return: between [0, 1]
    """
    n = 10.0
    x_num = np.sin(i/n)
    y_num = np.cos(i/n)
    z_num = x_num / 2 + y_num / 4 - x_num * y_num
    # z_num = 0.2 * np.sin(x_num) + 0.1 * y_num ** 2 + x_num/2
    return x_num, y_num, z_num


def compare_graphs(func_hat, func_real, compare: bool =True):
    alg_dataset = ALGDataset(y_func)
    x, y, z, z_real = [], [], [], []
    for obs_xy, obs_y in alg_dataset:
        obs_xy = torch.unsqueeze(obs_xy, 0)
        z_val = func_hat(obs_xy)
        z_real.append(obs_y)
        z.append(z_val.detach().item())

    fig = go.Figure()  # showscale=False, colorbar_x=-0.07,
    x_axis = list(range(len(z_real)))
    # fig.add_trace(go.Scatter(x=x_axis, y=x, name='x', opacity=0.1))
    # fig.add_trace(go.Scatter(x=x_axis, y=y, name='y', opacity=0.1))
    fig.add_trace(go.Scatter(x=x_axis, y=z_real, name='z_real', opacity=1))
    fig.add_trace(go.Scatter(x=x_axis, y=z, name='z'))
    fig.show()


