import torch.nn as nn
import torch

class PDiscriminator(nn.Module):
    def __init__(self,
                 init_depth=256,
                 init_resolution_size=(8,5),
                 LReLU_negative_slope=0.2,
                 input_depth=1,
                 last_layer_size=1,
                 mini_batch_normalization=False,
                 scale_factor = 0.5,
                 normalization=True):
        super(PDiscriminator, self).__init__()

        self.init_depth = init_depth
        self.init_resolution_size = init_resolution_size
        self.scale_factor = scale_factor
        self.LReLU_negative_slope = LReLU_negative_slope
        self.input_depth = input_depth
        self.last_layer_size = last_layer_size
        self.mini_batch_normalziation = mini_batch_normalization
        self.kernel_size = 3
        self.padding = 1
        self.normalization = normalization

        self.depths = [init_depth]

        self.blocks = nn.ModuleList()

        self.init_last_linear_layer()
        self.init_last_block()

        self.alpha = 0

    def get_current_device(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")    
        return device

    def init_last_linear_layer(self):
        self.last_layer = nn.Sequential(
            nn.Linear(self.depths[0], self.last_layer_size),
        )

    def init_last_block(self):
        first_depth = self.init_depth
        if self.mini_batch_normalziation:
            first_depth += 1

        self.base_block = nn.ModuleList()

        self.base_block.append( # from RGB Layer
            nn.Sequential(
                nn.Conv2d(
                    in_channels=self.input_depth,
                    out_channels=first_depth,
                    kernel_size=1
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
            )
        )

        self.base_block.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels=first_depth,
                    out_channels=self.init_depth,
                    kernel_size=self.kernel_size,
                    padding=self.padding
                ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
            )
        )
        self.base_block.append(
            nn.Sequential(
                nn.Linear(
                    in_features=self.init_depth * self.init_resolution_size[0] * self.init_resolution_size[1],
                    out_features=self.init_depth,
                    ),
                nn.LeakyReLU(negative_slope=self.LReLU_negative_slope),
            )
        )

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
            nn.Upsample(scale_factor=self.scale_factor, mode='nearest') # downsampling
        ))

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

    def downsampling(self, z):
        downsample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest') # downsampling
        return downsample(z)

    def reshape(self, z):
        if len(z.size()) == 4:
            size = z.size()[1:]
        else:
            size = z.size()
        reshape_size = 1
        for s in size:
            reshape_size *= s
        return z.view(-1, reshape_size)

    def forward(self, z):
        if self.alpha > 0 and self.alpha < 1 and len(self.blocks) > 0:
            y = self.downsampling(z)
            if len(self.blocks) == 1:
                y = self.base_block[0](y)
            else:
                y =  self.blocks[-2][0](y)
            bonding = True
        else:
            bonding = False

        for reversed_block_number, block in enumerate(reversed(self.blocks)):
            if reversed_block_number == 0:
                z = block[0](z)
            z = block[1](z)

            if bonding:
                bonding = False
                z = self.alpha * z + ( 1 - self.alpha) * y

        # TODO
        # minibatch standard deviation

        if len(self.blocks) == 0:
            z = self.base_block[0](z)

        z = self.base_block[1](z)
        z = self.reshape(z)
        z= self.base_block[2](z)
        out = self.last_layer(z)

        return out