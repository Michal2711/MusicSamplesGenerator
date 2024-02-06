import torch.nn as nn
import torch
import torch.nn.functional as F

from models.utils import miniBatchStdDev

class PDiscriminator(nn.Module):
    def __init__(self,
                 last_depth=256,
                 init_resolution_size=(8,5),
                 LReLU_negative_slope=0.2,
                 input_depth=1,
                 last_layer_size=1,
                 feature_size=1,
                 mini_batch_normalization=False,
                 scale_factor = 0.5,
                 normalization=False):
        r"""
        Args:
            init_depth (int): The initial depth (number of channels) for the model layers.
            init_resolution_size (tuple): The initial height and width of the input images.
            input_depth (int): The depth (number of channels) of the input images.
            last_layer_size (int): The size of the last layer before the output.
            scale_factor (float): The scaling factor for downscaling the image.
        """    

        super(PDiscriminator, self).__init__()

        self.last_depth = last_depth
        self.init_resolution_size = init_resolution_size
        self.scale_factor = scale_factor
        self.LReLU_negative_slope = LReLU_negative_slope
        self.input_depth = input_depth
        self.last_layer_size = last_layer_size
        self.feature_size = feature_size
        self.mini_batch_normalization = mini_batch_normalization
        self.kernel_size = 3
        self.padding = 1
        self.normalization = normalization

        self.depths = [last_depth]

        self.blocks = nn.ModuleList()
        self.init_last_block()

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

    def init_last_block(self):
        update_last_depth = self.last_depth
        if self.mini_batch_normalization:
            update_last_depth += 1

        self.base_block = nn.ModuleList()

        self.base_block.append( # fromRGB Layer
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.input_depth,
                    out_channels=self.last_depth,
                    kernel_size=1
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
            )
        )

        self.base_block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=update_last_depth,
                    out_channels=self.last_depth,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
                nn.Conv2d(
                    in_channels=update_last_depth,
                    out_channels=self.last_depth,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
                nn.Conv2d(
                    in_channels=self.last_depth,
                    out_channels=self.last_depth, 
                    kernel_size=(self.init_resolution_size[0], self.init_resolution_size[1]),
                    stride=1,
                    padding=0
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
            )
        )

        self.value_output = nn.Conv2d(
            in_channels=self.last_depth,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.classification_output = nn.Conv2d(
            in_channels=self.last_depth,
            out_channels=self.feature_size,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.base_block.apply(self.init_weights)
        self.value_output.apply(self.init_weights)
        self.classification_output.apply(self.init_weights)

    def create_block(self, last_depth, new_depth):
        block = nn.ModuleList()
        block.append(nn.Sequential( # from RGB Layer
            nn.Conv2d(
                in_channels=self.input_depth,
                out_channels=new_depth,
                kernel_size=1
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
        ))

        block.append(nn.Sequential(
            nn.Conv2d(
                in_channels=new_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
            nn.Conv2d(
                in_channels=new_depth,
                out_channels=last_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            ),
            nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
        ))

        block.apply(self.init_weights)
        return block

    def add_next_block(self, new_depth):
        last_depth = self.depths[-1]
        self.depths.append(new_depth)

        new_block = self.create_block(last_depth=last_depth, new_depth=new_depth)
        current_device = self.get_current_device()
        new_block = new_block.to(device=current_device)

        self.blocks.append(new_block)

    def set_alpha(self, new_alpha):
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError("New alpha must be in [0, 1]")
        
        if len(self.blocks) == 0:
            raise AttributeError("Can't set an alpha if only base block is defined")

        self.alpha = new_alpha

    def downsampling(self, z, size):
        return F.interpolate(z, size=size, mode='nearest')

    def reshape(self, z):
        if len(z.size()) == 4:
            size = z.size()[1:]
        else:
            size = z.size()
        reshape_size = 1
        for s in size:
            reshape_size *= s
        return z.view(-1, reshape_size)

    def forward(self, z, get_features=False):

        if len(self.blocks) == 0:
            z = self.base_block[0](z)

        if self.alpha < 1 and len(self.blocks) > 0:
            downsampled = self.downsampling(z, size=(z.shape[-2]//2, z.shape[-1]//2))
            if len(self.blocks) == 1:
                y = self.base_block[0](downsampled)
            else:
                y =  self.blocks[-2][0](downsampled)
            bonding = True
        else:
            bonding = False

        for reversed_block_number, block in enumerate(reversed(self.blocks)):
            if reversed_block_number == 0:
                z = block[0](z)
            z = block[1](z)
            z = self.downsampling(z, size=(z.shape[-2]//2, z.shape[-1]//2))

            if bonding:
                bonding = False
                z = self.alpha * z + (( 1.0 - self.alpha) * y)

        if self.mini_batch_normalization:
            z = miniBatchStdDev(z)

        z = self.base_block[1](z)

        value_output = self.value_output(z)
        
        if get_features:
            class_features = self.classification_output(z)
            return value_output, class_features 

        return value_output, None
