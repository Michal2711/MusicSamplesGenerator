import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self, height, width):
        # super(Discriminator, self).__init__()
        super().__init__()

        self.height = height
        self.width = width

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                    nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=2, padding=1),
                    nn.Dropout2d(0.25),
                    nn.LeakyReLU(0.2, inplace=True)
                ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        dummy_data = torch.zeros((1, 1, self.height, self.width))
        dummy_out = self.model(dummy_data)
        # print(dummy_out.shape)
        # flattened_size = dummy_out.view(-1).size(0)
        flattened_size = dummy_out.size(1) * dummy_out.size(2) * dummy_out.size(3)
        # print(f'flattened_size: {flattened_size}')
        self.adv_layer = nn.Sequential(
            nn.Linear(flattened_size, 1), 
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        # print(f'out before adv_layer: {out.size()}')
        validity = self.adv_layer(out)

        return validity
