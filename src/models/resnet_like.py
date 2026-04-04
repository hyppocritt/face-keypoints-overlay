from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):

        super().__init__()

        self.main_path = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.GELU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
        )

        if stride > 1 or out_channels > in_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                ),
                nn.BatchNorm2d(num_features=out_channels),
            )

        else:
            self.skip = nn.Identity()

        self.gelu = nn.GELU()

    def forward(self, x):

        x = self.main_path(x) + self.skip(x)
        x = self.gelu(x)

        return x


class FacePointsResNet(nn.Module):
    def __init__(self, input_size=224):

        super().__init__()

        self.input_size = input_size

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            ResBlock(32, 32),
            ResBlock(32, 64, 2),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 28, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.head(x)

        return x
