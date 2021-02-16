from alg_lit_module import *
from main_help_RNN import *
from alg_callbacks import *
from alg_data_module import *


if __name__ == '__main__':
    trainer = pl.Trainer(logger=NeptuneLogger(project_name="1919ars/NA-sandbox"),
                         max_epochs=10,
                         callbacks=[ALGCallback()])
    trainer.fit(model=alg_lit_module, datamodule=alg_data_module)

    # --- comparison to real function ---
    model = ALGLightningModule()
    model.load_state_dict(torch.load("example.ckpt"))
    compare_graphs(model, y_func)