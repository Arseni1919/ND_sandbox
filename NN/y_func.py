from IMPORTS import *


def y_func(a, b):
    """
    :param a: between [0, 1]
    :param b: between [0, 1]
    :return: between [0, 1]
    """
    mult = 50
    a *= mult
    b *= mult
    return np.sin(a / 5) * np.cos(b / 5) / 2 + 0.5  # - np.log(b)


def compare_graphs(func_hat, func_real, compare: bool =True):
    x = np.linspace(0.0, SCALE, num=50)
    y = np.linspace(0.0, SCALE, num=50)
    z_hat = np.zeros((len(x), len(y)))
    z_real = np.zeros((len(x), len(y)))
    scatter_dict_real = {'x': [], 'y': [], 'z': []}
    scatter_dict_hat = {'x': [], 'y': [], 'z': []}

    for i_indx, i in enumerate(x):
        for j_indx, j in enumerate(y):
            z_hat[i_indx, j_indx] = func_hat(torch.Tensor([[i, j]]).double()).item()
            z_real[i_indx, j_indx] = func_real(i, j)
            x_rund_num = random.uniform(0, SCALE)
            y_rund_num = random.uniform(0, SCALE)
            scatter_dict_real['x'].append(x_rund_num)
            scatter_dict_real['y'].append(y_rund_num)
            scatter_dict_real['z'].append(func_real(x_rund_num, y_rund_num))
            if compare:
                scatter_dict_hat['x'].append(x_rund_num)
                scatter_dict_hat['y'].append(y_rund_num)
                scatter_dict_hat['z'].append(
                    func_hat(torch.Tensor([[x_rund_num, y_rund_num]]).double()).item()
                )


    # fig = go.Figure(data=[go.Surface(z=z_hat),go.Surface(z=z_real, opacity=0.5)])# showscale=False, colorbar_x=-0.07,
    # fig.show()
    if compare:
        # fig1 = px.scatter_3d(scatter_dict_hat,
        #                     x='x', y='y', z='z', opacity=0.5, color='z', size_max=5)
        # fig2 = px.scatter_3d(scatter_dict_real,
        #                     x='x', y='y', z='z', opacity=0.1, color='z', size_max=5)
        # fig = go.Figure(data=[fig1, fig2])
        # fig.show()
        fig = go.Figure(data=[
            go.Surface(z=z_hat),
            go.Surface(z=z_real, opacity=0.1)
        ])# showscale=False, colorbar_x=-0.07,
        fig.show()
    else:
        fig = px.scatter_3d(scatter_dict_real,
                            x='x', y='y', z='z', opacity=0.1, color='z', size_max=5)
        fig.show()


def compare_stock_graphs(alg_lit_module):
    pass


if __name__ == '__main__':

    compare_graphs(func_hat=lambda x, y=0 : torch.Tensor([0]), func_real=y_func, compare=False)
