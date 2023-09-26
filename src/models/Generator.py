import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, height, width, latent_dim=32):
        # super(Generator, self).__init__()
        super().__init__()

        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.init_size = self.height // 4
        self.l1 = nn.Sequential(nn.Linear(self.latent_dim, 128 * self.init_size * (self.width // 4)))
        # self.len1 = nn.Sequential(nn.Linear(self.latent_dim, 4*self.height*self.width))

        # formula for output height and width for nn.ConvTranspose2d
        # h_out = (h_in) * stride - 2 * padding + kernel_height + output_padding-(kernel_height - 1)*(dilation - 1)
        # w_out = (w_in) * stride - 2 * padding + kernel_width + output_padding-(kernel_width - 1)*(dilation - 1)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

        # self.alter_conv_blocks = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        # )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.width // 4)
        # out = self.len1(z)
        # out = out.view(out.shape[0], 64, self.height/4, self.width/4)
        img = self.conv_blocks(out)
        return img
