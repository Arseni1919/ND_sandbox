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

# for i in alg_data_module.train_dataloader():
#     print(i)
#     print(len(i))
#     break


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

# fig = go.Figure()
# x_axis = np.arange(0, 1000)
# z = np.sin(x_axis) + np.cos(x_axis/2) + np.sin(x_axis/3) + np.cos(x_axis/4) + np.sin(x_axis/5)
# # + np.cos(x_axis/6) + np.sin(x_axis/7)
# fig.add_trace(go.Scatter(x=x_axis, y=z, name='z'))
# fig.show()

# import flash
# from pytorch_lightning import Trainer
# from flash.core.data import download_data
# from flash.text import TextClassificationData, TextClassifier
#
# download_data('https://pl-flash-data.s3.amazonaws.com/imdb.zip', 'data/')
#
# datamodule = TextClassificationData.from_files(
#                 train_file="data/imdb/train.csv",
#                 valid_file="data/imdb/valid.csv",
#                 test_file="data/imdb/test.csv",
#                 input="review",
#                 target="sentiment"
# )
#
# model = TextClassifier(num_classes = 2, backbone = 'roberta-base')
#
# trainer = flash.Trainer(max_epochs = 1)
# trainer.finetune(model, datamodule = datamodule)
# trainer.test()
# # Save Checkpoint
# trainer.save_checkpoint("text_class_model.pt")
#
# # Load Model From Checkpoint
# model = TextClassifier.load_from_checkpoint("text_class_model.pt")
# prediction = model.predict("This movie is great!")
# print(prediction)


# --------------

import flash
from flash.core.data import download_data
from flash.text import TextClassificationData, TextClassifier
# 1. Download the data
download_data("https://pl-flash-data.s3.amazonaws.com/imdb.zip", "data/")
# 2. Load the data
datamodule = TextClassificationData.from_files(
    train_file="data/imdb/train.csv",
    valid_file="data/imdb/valid.csv",
    test_file="data/imdb/test.csv",
    input="review",
    target="sentiment",
    batch_size=512
)
# 3. Build the model
model = TextClassifier(num_classes=datamodule.num_classes)
# 4. Create the trainer. Run once on data
trainer = flash.Trainer(max_epochs=1)
# 5. Fine-tune the model
trainer.finetune(model, datamodule=datamodule, strategy="freeze")
# 6. Test model
trainer.test()
# 7. Predict on new data
predictions = model.predict([
    "Turgid dialogue, feeble characterization - Harvey Keitel a judge?.",
    "I come from Bulgaria where it 's almost impossible to have a tornado."
    "This guy has done a great job with this movie!",
])
print(predictions)
# 8. Save the model!
trainer.save_checkpoint("text_classification_model.pt")





