import torch.nn as nn
import torch

class WeightedSum(nn.Module):
    def __init__(self, alpha=0.0):
        super(WeightedSum, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)

    def forward(self, x1, x2):
        return (1.0 - self.alpha) * x1 + self.alpha * x2

class PDiscriminator(nn.Module):
    def __init__(self,
                 init_depth=256,
                 init_resolution_size=(8,5),
                 LReLU_negative_slope=0.2,
                 input_depth=1,
                 decision_layer_size=1,
                 mini_batch_normalization=False,
                 scale_factor = 1/2,
                 normalization=True):
        super(PDiscriminator, self).__init__()

        self.init_depth = init_depth
        self.init_resolution_size = init_resolution_size
        self.scale_factor = scale_factor
        self.LReLU_negative_slope = LReLU_negative_slope
        self.input_depth = input_depth
        self.decision_layer_size = decision_layer_size
        self.mini_batch_normalziation = mini_batch_normalization
        self.kernel_size = 3
        self.padding = 1
        self.normalization = normalization

        self.depths = [init_depth]

        self.LeakyRelu = nn.LeakyReLU(negative_slope=LReLU_negative_slope)
        self.scaleLayers = nn.Modulelist()
        self.mergeLayers = nn.Modulelist()
        
        self.groupScaleZero = nn.ModuleList()
        self.fromRGBLayers = nn.ModuleList()

        self.initDecisionLayer()
        self.initScale0Layer()

        self.alpha = 0

    def initDecisionLayer(self):
        self.decision_layer = nn.Linear(self.depths[0], 
                                        self.decision_layer_size)
        
    def initScale0Layer(self):
        # minibatch standard deviation
        dimEntryScale0 = self.init_depth
        if self.mini_batch_normalziation:
            dimEntryScale0 += 1

        self.fromRGBLayers.append(
            nn.Conv2d(
                in_channels=self.input_depth,
                out_channels=self.init_depth,
                kernel_size=1
            )
        )

        self.groupScaleZero.append(
            nn.Conv2d(
                in_channels=dimEntryScale0,
                out_channels=self.init_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )

        self.groupScaleZero.append(
            nn.Linear(
                in_channels=self.init_depth * self.init_resolution_size[0] * self.init_resolution_size[1],
                out_channels=self.init_depth,
            )
        )

    def addScale(self, new_depth):
        last_depth = self.depths[-1]
        self.depths.append(new_depth)

        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(
            nn.Conv2d(
                in_channels=new_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )    

        self.scaleLayers[-1].append(
            nn.Conv2d(
                in_channels=new_depth,
                out_channels=last_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )

        self.fromRGBLayers[-1].append(
            nn.Conv2d(
                in_channels=self.input_depth,
                out_channels=new_depth,
                kernel_size=1
            )
        )
    
    def set_alpha(self, new_alpha):
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError("New alpha must be in [0, 1]")
        
        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only init scale is defined")

        self.alpha = new_alpha

    def downscale(self, z):
        return nn.Upsample(z, scale_factor=self.scale_factor, mode='nearest')
    
    def forward(self, z, getFeature=False):
        
        if self.alpha > 0 and len(self.fromRGBLayers) > 1:
            y = self.downscale(z)
            y =  self.LeakyRelu(self.fromRGBLayers[-2](y))

        z = self.LeakyRelu(self.fromRGBLayers[-1](z))

        mergeLayer = self.alpha > 0 and len(self.scaleLayers) > 1
        shift = len(self.fromRGBLayers) - 2

        for _, groupLayer in enumerate(reversed(self.scaleLayers)):
            for layer in groupLayer:
                z = self.LeakyRelu(layer(z)) # na pewno?

            z = self.downscale(z)

            if mergeLayer:
                mergeLayer = False
                z = self.alpha * y + ( 1 - self.alpha) * z

            shift -= 1

        # Scale 0
        # minibatch standard deviation
        # TODO
        # if self.mini_batch_normalziation:
            # z = miniBatchStdDev(z)
            
        z = self.LeakyRelu(self.groupScaleZero[0](z))

        z_lin = self.groupScaleZero[1](z)
        z = self.LeakyRelu(z_lin)

        out = self.decision_layer(z)

        if not getFeature:
            return out
        
        return out, z_lin