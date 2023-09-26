import torch.nn as nn
import torch
import torch.nn.functional as F

class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, z, epsilon=1e-8):
        return z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + epsilon)


class PGenerator2(nn.Module):
    def __init__(self, 
                 init_depth=256, 
                 init_resolution_size=(8, 5),
                 latent_dim=100,
                 scale_factor=2,
                 output_depth=1,
                 LReLU_negative_slope=0.2,
                 toRGBActivation=None,
                 normalization=True):
        super(PGenerator2, self).__init__()

        self.init_depth = init_depth # number of channels for first resolution
        self.output_depth = output_depth # number of output channels
        self.init_resolution_size = init_resolution_size # 
        self.scale_factor=scale_factor # parameter for upsampling
        self.latent_dim = latent_dim
        self.LReLU_negative_slope = LReLU_negative_slope
        self.kernel_size = 3
        self.padding = 1
        self.normalization = normalization
        self.toRGBActivation=toRGBActivation # funkcja aktywacji dla warsty toRGB, jesli None to uzyjemy identity

        self.depths = [init_depth]

        self.blocks = nn.ModuleList()

        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        self.init_first_linear_layer()
        self.init_first_block()

        self.alpha = 0


    def init_first_linear_layer(self):
        self.l1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.init_depth * self.init_resolution_size[0] * self.init_resolution_size[1]),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
        )

    def init_first_block(self):
        self.base_block = nn.ModuleList()

        self.base_block.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.depths[0],
                    out_channels=self.depths[0],
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
            )
        )

        self.base_block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.depths[0],
                    out_channels=self.output_depth,
                    kernel_size=1,
                )
            )
        )

    def create_block(self, last_depth, new_depth):
        block = nn.ModuleList()
        block.append(nn.Sequential(
            nn.Upsample(scale_factor=self.scale_factor, mode='nearest'),
            nn.ConvTranspose2d(
                in_channels=last_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
            # self.normalizationLayer(),
            nn.ConvTranspose2d(
                in_channels=new_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
        ))

        block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=new_depth,
                    out_channels=self.output_depth,
                    kernel_size=1,
                ),
            )
        )
        return block

    def add_next_block(self, new_depth):
        last_depth = self.depths[-1]
        self.depths.append(new_depth)

        self.blocks.append(self.create_block(last_depth=last_depth, new_depth=new_depth))

    def get_output_size(self):
        if type(self.init_resolution_size) == tuple:
            size_h = int(self.init_resolution_size[0] * (2**(len(self.blocks))))
            size_w = int(self.init_resolution_size[1] * (2**(len(self.blocks))))
            return (size_h, size_w)
        else:
            size = self.init_resolution_size * (2**(len(self.blocks)))
            return (size, size)

    def set_alpha(self, new_alpha):
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError("New alpha must be in [0, 1]")
        
        if len(self.blocks) == 0:
            raise AttributeError("Can't set an alpha if only base block is defined")

        self.alpha = new_alpha

    def transform_to_init_resolution_shape(self, z):
        return z.view(z.size(0), -1, self.init_resolution_size[0], self.init_resolution_size[1])
    
    def forward(self, z):
        if self.normalizationLayer is not None:
            z = self.normalizationLayer(z)

        z = self.l1(z)
        print(z.shape)
        z = self.transform_to_init_resolution_shape(z)

        print(z.shape)

        z = self.normalizationLayer(z)

        z = self.base_block[0](z)
        if self.normalizationLayer is not None:
            z = self.normalizationLayer(z)

        if self.alpha == 0 and len(self.blocks) == 0:
            y = self.base_block[1](z) # RGB Layer
            print(f'return: {y.shape}')
            return y

        if self.alpha > 0 and len(self.blocks) == 1:
            y = self.base_block[1](z) # RGB Layer
            print(f'y shape: {y.shape}')

        block_number = 0
        for block_number, block in enumerate(self.blocks, 0):
            print(f'block number: {block_number} with shape input: {z.shape}')
            z = block[0](z)
            # TODO tu jeszcze przed drugim conv powinno byc norm
            if self.normalizationLayer is not None:
                z = self.normalizationLayer(z)

            if self.alpha == 0 and block_number == (len(self.blocks)-1):
                print(f'first block output before RGB shape: {z.shape}')
                z = block[1](z)
                print(f'first block output after RGB shape: {z.shape}')

            if self.alpha > 0:
                if block_number == (len(self.blocks)-2):
                    y = block[1](z)
                elif block_number == (len(self.blocks)-1):
                    z = block[1](z)

        if self.alpha > 0:
            z = (1.0 - self.alpha * y) + self.alpha * z

        if self.toRGBActivation is not None:
            z = self.toRGBActivation(z)

        return z
