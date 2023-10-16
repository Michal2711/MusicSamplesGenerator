import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, height, width, transformation_type, latent_dim=32):
        super().__init__()
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        
        if transformation_type == 'mel':
            self.init_size = 16
            self.init_width = self.width // 16
        else:
            self.init_size = self.height // 16
            self.init_width = self.width // 16

        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 256 * self.init_size * self.init_width))
        self.deconv_blocks = nn.ModuleList()
        
        in_channels, out_channels = 256, 128
        for i in range(4):
            if i == 0 or i == 1:
                self.deconv_blocks.append(self._block(in_channels, out_channels, kernel_size=3, stride=2))
            else:
                self.deconv_blocks.append(self._block(in_channels, out_channels, kernel_size=2, stride=2))
            in_channels = out_channels
            out_channels //= 2

        self.deconv_blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels, 1, kernel_size=(15,2), stride=1, padding=1),
                nn.Tanh()
            )
        )

    def _block(self, in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.BatchNorm2d(out_channels, 0.8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 256, self.init_size, self.init_width)

        for block in self.deconv_blocks:
            out = block(out)

        return out
