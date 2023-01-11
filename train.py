import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.model_summary import summarize

import torch
from model import GestureRecognitionModel
from dataset import GestureRecognitionDataModule

import pandas as pd

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# hyperparameter
cfg = {
    'fps' : 30,
    'img_size' : 128,
    'epochs' : 10,
    'lr' : 3e-4,
    'batch_size' : 4,
    'seed' : 41,
    'num_workers' : 4,
    'resume_from' : '',
    'train_df_path' : './train.csv',
    'test_df_path' : './test.csv',
    'ckpt_path' : './checkpoints'
}

cfg = AttrDict(cfg)


# define model
model = GestureRecognitionModel(config = cfg)

if cfg.resume_from != '' :
    model.load_state_dict(torch.load(cfg.resum_from))
print(summarize(model, max_depth = 2))

# set datamodule
train_df = pd.read_csv(cfg.train_df_path)
test_df = pd.read_csv(cfg.test_df_path)

datamodule = GestureRecognitionDataModule(config = cfg, 
                             train_df = train_df, 
                             test_df = test_df,
                             collate_fn = None)

# checkpoint
checkpoint = ModelCheckpoint(monitor = 'f1 score(macro)',
                             mode = 'max',
                             save_top_k = 3,
                             save_last = True,
                             filename = '{epoch}-{step}-{f1 score(macro)}'
                            )

# train model
trainer = Trainer(accelerator = 'gpu', devices = 1,
                  logger = TensorBoardLogger(save_dir = cfg.ckpt_path),
                  enable_model_summary = False,
                  callbacks = [checkpoint]
                  )
trainer.fit(model = model, datamodule = datamodule)

