import torch.nn as nn
import torch
import torch.nn.functional as F

class PGenerator(nn.Module):
    def __init__(self, 
                 init_depth=256, 
                 init_resolution_size=(8, 5),
                 latent_dim=256,
                 scale_factor=2,
                 output_depth=1,
                 LReLU_negative_slope=0.2,
                 normalization=True,
                 feature_size=0):
        r"""
        Args:
            init_depth (int): Initial depth (number of channels) for the model layers.
            init_resolution_size (tuple): Initial height and width of the output images for first block.
            scale_factor (float): The scaling factor for upscaling the image.
            output_depth (int): Depth (number of channels) of the output images.
        """

        super(PGenerator, self).__init__()

        self.init_depth = init_depth
        self.output_depth = output_depth
        self.init_resolution_size = init_resolution_size
        self.scale_factor=scale_factor
        self.latent_dim = latent_dim
        self.LReLU_negative_slope = LReLU_negative_slope
        self.kernel_size = 3
        self.padding = 1
        self.normalization = normalization
        self.transposed = True

        self.depths = [init_depth]

        self.blocks = nn.ModuleList()
        self.feature_size = feature_size

        self.init_first_block()
        self.alpha = 0

    def get_current_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    
        return device

    def init_weights(self, m):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            m.bias.data.fill_(0.0)

    def init_first_block(self):
        self.base_block = nn.ModuleList()

        self.base_block.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.latent_dim + self.feature_size,
                    out_channels=self.depths[0],
                    kernel_size=(self.init_resolution_size[0], self.init_resolution_size[1]),
                    stride=1,
                    padding=0
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
                nn.ConvTranspose2d(
                    in_channels=self.depths[0],
                    out_channels=self.depths[0],
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
                nn.ConvTranspose2d(
                    in_channels=self.depths[0],
                    out_channels=self.depths[0],
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
            )
        )

        self.base_block.append( # toRGB
            nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=self.depths[0],
                    out_channels=self.output_depth,
                    kernel_size=1,
                )
            )
        )

        self.base_block.apply(self.init_weights)

    def create_block(self, last_depth, new_depth):
        block = nn.ModuleList()
        block.append(nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=last_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
            nn.ConvTranspose2d(
                in_channels=new_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
        ))

        block.append( # toRGB
            nn.Sequential(                
                nn.ConvTranspose2d(
                    in_channels=new_depth,
                    out_channels=self.output_depth,
                    kernel_size=1,
                ),
            )
        )

        block.apply(self.init_weights)
        return block

    def add_next_block(self, new_depth):
        last_depth = self.depths[-1]
        self.depths.append(new_depth)

        new_block = self.create_block(last_depth=last_depth, new_depth=new_depth)
        
        current_device = self.get_current_device()
        new_block = new_block.to(device=current_device)

        self.blocks.append(new_block)

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

    def upsampling(self, z, size):
        return F.interpolate(z, size=size, mode='nearest')
    
    def forward(self, z, feature_vector=None):

        block_out = self.base_block[0](z)

        if len(self.blocks) == 0:
            base_RGB = self.base_block[1](block_out)
            return base_RGB

        for block_number, block in enumerate(self.blocks, 0):

            upsampled = self.upsampling(block_out, size=(block_out.shape[-2]*2, block_out.shape[-1]*2))
            block_out = block[0](upsampled)

            if block_number == len(self.blocks)-1:
                old_RGB = self.blocks[block_number-1][1](upsampled)
                last_block_RGB = block[1](block_out)

        if self.alpha < 1:
            out = ((1.0 - self.alpha) * old_RGB) + self.alpha * last_block_RGB
        else:
            out = last_block_RGB
        
        return out
