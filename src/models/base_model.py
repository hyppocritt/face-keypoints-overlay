import torch
from torch import nn 


class FacePointsModel(nn.Module):

    def __init__(self, input_size=224):

        super().__init__()

        self.input_size = input_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d(4)
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 28)
        )
    
    def forward(self, x):

        x = self.features(x)
        x = self.head(x)

        return x

