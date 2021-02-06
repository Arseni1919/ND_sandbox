from IMPORTS import *


def y_func(a, b):
    return np.sin(a / 5) * np.cos(b / 5) / 2 + 0.5  # - np.log(b)


if __name__ == '__main__':

    x = np.linspace(0.0, SCALE, num=100)
    y = np.linspace(0.0, SCALE, num=100)
    z = np.zeros((len(x), len(y)))

    for i_indx, i in enumerate(x):
        for j_indx, j in enumerate(y):
            #         z[i,j] = np.log(i) + np.log(j)
            z[i_indx, j_indx] = y_func(i, j)
    #         z[i_indx,j_indx] = np.sin(10*(i**2+j**2))/10.0

    fig = go.Figure(
        data=go.Surface(z=z)
    )
    fig.show()
