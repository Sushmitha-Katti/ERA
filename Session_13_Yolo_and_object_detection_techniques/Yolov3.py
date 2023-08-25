import pytorch_lightning as pl
from torch.optim.lr_scheduler import OneCycleLR

import torch
import config
import torch.nn as nn
import torch.optim as optim
import utils

class Yolov3(pl.LightningModule):
    def __init__(self, model, scaled_anchors, train_loader = None, loss_fn = None, in_channels=3, num_classes=20, threshold = 0.5,  learning_rate = 0.0001):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_fn = loss_fn
        self.scaled_anchors = scaled_anchors
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.model = model       
        
    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
     
        x, y = batch
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2],
        )

        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )
        self.log('train_loss', loss,prog_bar=True)

        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y0, y1, y2 = (
            y[0],
            y[1],
            y[2],
        )

        with torch.cuda.amp.autocast():
            out = self(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )
        self.log('val_loss', loss,prog_bar=True)
        return {'val_loss': loss}
     

    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        
    def on_train_epoch_end(self,):
        if self.train_loader:
            check_class_accuracy(self, train_loader, threshold=config.CONF_THRESHOLD)



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1E-3, weight_decay=1e-4)
        print("Learning Rate: ", self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr = 1E-3,   # Suggested LR
                                                epochs= self.trainer.max_epochs,
                                                steps_per_epoch= len(self.train_loader) if self.train_loader else 32,
                                                pct_start=5/self.trainer.max_epochs if self.trainer.max_epochs >= 5 else 1,
                                                div_factor=100,
                                                three_phase=False,
                                                final_div_factor=100,
                                                anneal_strategy="linear"
                                                       )
        return [ optimizer],[ {'scheduler': scheduler,'interval': 'step', 'frequency': 1}]