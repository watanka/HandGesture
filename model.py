import pytorch_lightning as pl
import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from sklearn.metrics import f1_score

from abc import ABC, abstractmethod

class BaseSystem(pl.LightningModule, ABC) :
    '''모델 종류가 다양해질 경우 적용'''
    pass


class Conv3DModel(nn.Module) :

    def __init__(self, num_classes = 5) :
        super().__init__()

        self.feature_extract = nn.Sequential(
            nn.Conv3d(3, 8, (3,3,3)),
            nn.ReLU(),
            nn.BatchNorm3d(8), 
            nn.MaxPool3d(2),
            nn.Conv3d(8, 32, (2,2,2)),
            nn.ReLU(),
            nn.BatchNorm3d(32),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, (2,2,2)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, (2,2,2)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((1,7,7))
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x) :
        batch_size = x.shape[0]
        res = self.feature_extract(x)
        res = res.view(batch_size, -1)
        res = self.classifier(res)
        return res




class GestureRecognitionModel(pl.LightningModule) :

    def __init__(self, config, model = Conv3DModel) : 
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = Conv3DModel()

    def training_step(self, batch, batch_index) :
        videos, labels = batch
        output = self.model(videos)
        loss = F.cross_entropy(output, labels)
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx) :
        return self._eval_step(batch)
    
    def configure_optimizers(self) :
        optimizer = torch.optim.Adam(self.parameters(), lr = self.config.lr )
        return optimizer
    
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx) :
        optimizer.zero_grad(set_to_none = True)

    def _eval_step(self, batch) :

        videos, labels = batch

        correct = 0
        total = 0
        preds, trues = [], []
        val_loss = []

        with torch.no_grad() :
            logit = self.model(videos)
            loss = F.cross_entropy(logit, labels)

            val_loss.append(loss.item())
            preds += logit.argmax(1).detach().cpu().numpy().tolist()
            trues += labels.detach().cpu().numpy().tolist()

        
        val_loss = np.mean(val_loss)
        val_score = f1_score(trues, preds, average = 'macro')

        return val_loss, val_score

    def validation_epoch_end(self, outputs) :
        val_loss, val_score = zip(*outputs)
        self.log('f1 score(macro)', np.mean(val_score), sync_dist = True)
        self.log('val loss', np.mean(val_loss), sync_dist = True)


