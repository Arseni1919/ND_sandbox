import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import pandas as pd
import plotly.graph_objects as go
import cufflinks as cf
import sklearn
from sklearn import preprocessing
# import chart_studio.plotly as py
import seaborn as sns
import plotly.express as px
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import sys
# sys.path.append("/home/arseni1919/PycharmProjects/NT_sandbox")


import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from collections import namedtuple, deque
import gym
from pytorch_lightning.loggers.neptune import NeptuneLogger
import random

toy_data = {'a': 1, 'b': 2}

LR = 1e-2  # learning rate
REPLAY_SIZE = 1600  # 10048
BATCH_SIZE = 16
SCALE = 1.0
WINDOW_TO_LOOK_BACK = 13
