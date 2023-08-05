import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader, random_split


from utils import Cifar10SearchDataset


PATH_DATASETS = os.environ.get("CIFAR_DATASET", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 512 if AVAIL_GPUS else 64


# Basic Block
class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Custom Block
class CustomBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomBlock, self).__init__()

        self.inner_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.res_block = BasicBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.inner_layer(x)
        r = self.res_block(x)

        out = x + r

        return out


class CustomResNet(LightningModule):
    def __init__(self, num_classes=10, data_dir=PATH_DATASETS, learning_rate = 0.03 ):


        super(CustomResNet, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes =num_classes,
        self.learning_rate = learning_rate
        self.save_hyperparameters()


        # set the data directory path
        self.data_dir = data_dir

        self.num_classes = 10
        means = [0.4914, 0.4822, 0.4465]
        stds = [0.2470, 0.2435, 0.2616]

        # Transforms
        self.train_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                A.PadIfNeeded(min_height=40, min_width=40, always_apply=True),
                A.RandomCrop(height=32, width=32, always_apply=True),
                A.HorizontalFlip(),
                A.CoarseDropout(max_holes=1, max_height=8, max_width=8, min_holes=1, min_height=8, min_width=8, fill_value=means),
                ToTensorV2(),
            ]
        )

        self.test_transforms = A.Compose(
            [
                A.Normalize(mean=means, std=stds, always_apply=True),
                ToTensorV2(),
            ]
        )
        


        # Model
        self.prep_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )


        self.layer_1 = CustomBlock(in_channels=64, out_channels=128)

        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.layer_3 = CustomBlock(in_channels=256, out_channels=512)

        self.max_pool = nn.Sequential(nn.MaxPool2d(kernel_size=4))

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.prep_layer(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


    ##########################################
    # Train Test Hooks
    ###########################################
    def validation_step(self, batch, batch_idx):

        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc=  accuracy(preds, y, task= 'multiclass', num_classes=self.num_classes)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)



        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.03, weight_decay=1e-4)
        print("Learning Rate: ", self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                max_lr = self.learning_rate ,   # Suggested LR
                                                epochs= self.trainer.max_epochs,
                                                steps_per_epoch=len(self.train_dataloader()),
                                                pct_start=5/self.trainer.max_epochs,
                                                div_factor=100,
                                                three_phase=False,
                                                final_div_factor=100,
                                                anneal_strategy="linear"
                                                )
        return {'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Adjust the LR on each step
                'frequency': 1,      # Adjust LR every step
            }
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc =  accuracy(preds, y, task= 'multiclass', num_classes=self.num_classes)
        self.log('train_loss', loss,  prog_bar=True)
        self.log("train_acc", acc,  prog_bar=True)
        # Access the current learning rate
        current_lr = self.optimizers().param_groups[0]['lr']

        # Log the learning rate
        self.log('learning_rate', current_lr)
        return loss


    ##### DATA Hooks #########

    def prepare_data(self):
      # download
        Cifar10SearchDataset(root=self.data_dir, train=True,
                                        download=True, transform=self.train_transforms)
        Cifar10SearchDataset(root=self.data_dir, train=False,
                                        download=True, transform=self.test_transforms)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_full = Cifar10SearchDataset(self.data_dir, train=True,
                                          download=True, transform=self.train_transforms)
            self.cifar_train, self.cifar_val = random_split(self.cifar_full, [0.8, 0.2])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = Cifar10SearchDataset(root=self.data_dir, train=False,
                                        download=True, transform=self.test_transforms)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=BATCH_SIZE, num_workers=os.cpu_count(),pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=BATCH_SIZE, num_workers=os.cpu_count(),pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=BATCH_SIZE, num_workers=os.cpu_count(),pin_memory=True)
