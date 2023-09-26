import torch.nn as nn

class Discriminator3(nn.Module):
    def __init__(self, height, width, transformation_type):
        super().__init__()

        self.height = height
        self.width = width
        
        if transformation_type == 'mel':
            self.init_size = 16
            self.init_width = self.width // 16
        else:
            self.init_size = self.height // 16
            self.init_width = self.width // 16

        self.start = nn.Sequential(
            nn.ConvTranspose2d(1, 16, kernel_size=(15,2), stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Create a list to hold all our conv layers
        self.conv_blocks = nn.ModuleList()
        
        in_channels, out_channels = 16, 32
        for i in range(4):
            if i == 2 or i == 3:
                self.conv_blocks.append(self._block(in_channels, out_channels, kernel_size=3, stride=2))
            else:
                self.conv_blocks.append(self._block(in_channels, out_channels, kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels *= 2
        
        # Flatten and fully connected layer to get a probability
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(256 * self.init_size * self.init_width, 1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img):
        out = self.start(img)

        # print(out.size())

        for block in self.conv_blocks:
            out = block(out)
            # print(out.size())
        out = self.flatten(out)
        # print(out.size())
        out = self.fc(out)
        # print(out.size())

        return out