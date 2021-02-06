# sys.path.append("/home/arseni1919/PycharmProjects/NT_sandbox")
# sys.path.append("/Users/arseniperchik/PycharmProjects/NT_sandbox")
# print(sys.path)
# from IMPORTS import *

from alg_lit_module import *
from y_func import *
from alg_callbacks import *
from alg_dataset import *
from alg_data_module import *
from config import API_KEY


if __name__ == '__main__':

    # alg_dataset = ALGDataset(y_func)
    # alg_data_module = ALGDataModule(alg_dataset)
    # alg_lit_module = ALGLightningModule(n_input=2, n_output=1)

    alg_dataset = STOCKSDataset()
    alg_data_module = ALGDataModule(alg_dataset)
    alg_lit_module = ALGLightningModule(n_input=104, n_output=1)

    trainer = pl.Trainer(logger=NeptuneLogger(project_name="1919ars/NA-sandbox",
                                              api_key=API_KEY,
                                              # gpus=1
                                              ), max_epochs=20, callbacks=[ALGCallback()])

    trainer.fit(model=alg_lit_module, datamodule=alg_data_module)

    # --- comparison to real function ---
    alg_lit_module.load_state_dict(torch.load("example.ckpt"))
    # compare_graphs(alg_lit_module, y_func)
    compare_stock_graphs(alg_lit_module)
