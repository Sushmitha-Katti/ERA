import torch.nn as nn
import torch.nn.functional as F

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        self.conv1 = nn.Sequential(                      
            
        nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),#Op_size = 26, RF = 3
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Dropout2d(0.01),

        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),#Op_size = 24, RF = 5
        nn.ReLU(),
        nn.BatchNorm2d(10),
        nn.Dropout2d(0.01),

        nn.Conv2d(in_channels=10, out_channels=12, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 10, RF = 7
        nn.ReLU(),
        nn.BatchNorm2d(12),
        nn.Dropout2d(0.01),

        nn.MaxPool2d(kernel_size=(2,2)), #Op_size = 12, RF = 8

        nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False), #Op_size = 8, RF = 12
        nn.ReLU(),
        nn.BatchNorm2d(14),
        nn.Dropout2d(0.01),

        nn.Conv2d(in_channels=14, out_channels=14, kernel_size=(3, 3), padding=1, bias=False),#Op_size = 6, RF = 16
        nn.ReLU(),
        nn.BatchNorm2d(14),
        nn.Dropout(0.01),

        nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),#Op_size = 6, RF = 20
        nn.ReLU(),
        nn.BatchNorm2d(16),
        nn.Dropout(0.01),

        nn.AvgPool2d(kernel_size=6),


        nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)#Op_size = 1, RF = 24
        ) 

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
