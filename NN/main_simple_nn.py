from IMPORTS import *
from alg_lit_module import *
from y_func import y_func
from alg_callbacks import *
# sys.path.append("/home/arseni1919/PycharmProjects/NT_sandbox")
# sys.path.append("/Users/arseniperchik/PycharmProjects/NT_sandbox")
print(sys.path)


def compare_graphs(func_hat, func_real):
    x = np.linspace(0.0, SCALE, num=100)
    y = np.linspace(0.0, SCALE, num=100)
    z_hat = np.zeros((len(x), len(y)))
    z_real = np.zeros((len(x), len(y)))

    for i_indx, i in enumerate(x):
        for j_indx, j in enumerate(y):
            z_hat[i_indx, j_indx] = func_hat(torch.Tensor([[i / 100.0, j / 100.0]])).item()
            z_real[i_indx, j_indx] = func_real(i, j)

    fig = go.Figure(
        data=[
            go.Surface(z=z_hat),
            go.Surface(z=z_real, opacity=0.5)
        ]
    )
    # showscale=False, colorbar_x=-0.07,
    fig.show()


if __name__ == '__main__':
    PARAMS = {'LR': LR}
    neptune_logger = NeptuneLogger(project_name="1919ars/NA-sandbox")

    trainer = pl.Trainer(logger=neptune_logger, max_epochs=2, callbacks=[ALGCallback()])
    # trainer.fit(model=alg_module, datamodule=data_module)
    trainer.fit(model=alg_lit_module)

    # --- comparison to real function ---
    model = ALGLightningModule()
    model.load_state_dict(torch.load("example.ckpt"))
    compare_graphs(model, y_func)
