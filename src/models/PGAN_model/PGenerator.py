import torch.nn as nn
import torch
import torch.nn.functional as F

class NormalizationLayer(nn.Module):

    def __init__(self):
        super(NormalizationLayer, self).__init__()

    def forward(self, z, epsilon=1e-8):
        return z / torch.sqrt(torch.mean(z ** 2, dim=1, keepdim=True) + 1e-8)


class PGenerator(nn.Module):
    def __init__(self, 
                 init_depth=256, 
                 init_resolution_size=(8, 5),
                 latent_dim=100,
                 scale_factor=2,
                 output_depth=1,
                 LReLU_negative_slope=0.2,
                 toRGBActivation=None,
                 normalization=True):
        super(PGenerator, self).__init__()

        self.init_depth = init_depth # number of channels for first resolution
        self.output_depth = output_depth # number of output channels
        self.init_resolution_size = init_resolution_size # 
        self.scale_factor=scale_factor
        self.latent_dim = latent_dim
        self.LReLU_negative_slope = LReLU_negative_slope # parametr LReLU
        self.kernel_size = 3
        self.padding = 1
        self.normalization = normalization
        self.toRGBActivation=toRGBActivation # funkcja aktywacji dla warsty toRGB, jesli None to uzyjemy identity

        self.depths = [init_depth]

        self.LeakyRelu = nn.LeakyReLU(negative_slope=self.LReLU_negative_slope)
        self.scaleLayers = nn.ModuleList()
        self.toRGBLayers = nn.ModuleList()

        self.normalizationLayer = None
        if normalization:
            self.normalizationLayer = NormalizationLayer()

        self.initFirstLinearLayer()
        self.initScale0Layer()

        self.alpha = 0

    def initFirstLinearLayer(self):
        self.l1 = nn.Linear(self.latent_dim, self.depths[0] * self.init_resolution_size[0] * self.init_resolution_size[1])

    def initScale0Layer(self):
        self.groupScale0 = nn.ModuleList()

        self.groupScale0.append(
            nn.ConvTranspose2d(
                in_channels=self.depths[0],
                out_channels=self.depths[0],
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )

        self.toRGBLayers.append(
            nn.Conv2d(
                in_channels=self.depths[0],
                out_channels=self.output_depth,
                kernel_size=1,
            )
        )

    def getOutputSize(self):
        # maybe -1 with 2**(here)
        if type(self.init_resolution_size) == tuple:
            size_h = int(self.init_resolution_size[0] * (2**(len(self.toRGBLayers))))
            size_w = int(self.init_resolution_size[1] * (2**(len(self.toRGBLayers))))
            return (size_h, size_w)
        else:
            size = self.init_resolution_size * (2**(len(self.toRGBLayers)))
            return (size, size)

    def addScale(self, new_depth):
        r"""
            Args:
                - depths - depths for each convolutional layer
        """
        last_depth = self.depths[-1]
        self.depths.append(new_depth)

        self.scaleLayers.append(nn.ModuleList())

        self.scaleLayers[-1].append(
            nn.ConvTranspose2d(
                in_channels=last_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )
        self.scaleLayers[-1].append(
            nn.ConvTranspose2d(
                in_channels=new_depth,
                out_channels=new_depth,
                kernel_size=self.kernel_size,
                padding=self.padding
            )
        )

        self.toRGBLayers.append(
            nn.Conv2d(
                in_channels=new_depth,
                out_channels=self.output_depth,
                kernel_size=1,
            )
        )

    def set_alpha(self, new_alpha):
        if new_alpha < 0 or new_alpha > 1:
            raise ValueError("New alpha must be in [0, 1]")
        
        if not self.toRGBLayers:
            raise AttributeError("Can't set an alpha layer if only init scale is defined")

        self.alpha = new_alpha

    def upscale(self, z):
        return nn.Upsample(z, scale_factor=self.scale_factor, mode='nearest')

    def change_init_view(self, z):
        return z.view(z.size(0), -1, self.init_resolution_size[0], self.init_resolution_size[1])
    
    def forward(self, z, test_all_scales=False):

        output = []

        if self.normalizationLayer is not None:
            z = self.normalizationLayer(z)

        z = self.LeakyRelu(self.l1(z))
        z = self.change_init_view(z)

        z = self.normalizationLayer(z)

        for convLayer in self.groupScale0:
            z = self.LeakyRelu(convLayer(z))
            if self.normalizationLayer is not None:
                z = self.normalizationLayer(z)

        if self.alpha > 0 and len(self.scaleLayers) == 1:
            y = self.toRGBLayers[-2](z)
            y = self.upscale(y)

        if test_all_scales:
            output.append(self.toRGBLayers[0](z))

        scale = 0
        for scale, layerGroup in enumerate(self.scaleLayers, 0):
            z = self.upscale(z)
            for convLayer in layerGroup:
                z = self.LeakyRelu(convLayer(z))
                if self.normalization is not None:
                    z = self.normalizationLayer(z)

            if test_all_scales and scale <= len(self.scaleLayers) - 2:
                output.append(self.toRGBLayers[scale + 1](z))

            if self.alpha > 0 and scale == (len(self.scaleLayers) -2 ):
                y = self.toRGBLayers[-2](z)
                y = self.upscale(y)

        z = self.toRGBLayers[-1](z)

        if self.alpha > 0:
            z = self.alpha * y + (1.0 - self.alpha) * z

        if self.toRGBActivation is not None:
            z = self.toRGBActivation(z)

        if test_all_scales and scale !=0:
            output.append(z)

        if test_all_scales:
            return output
        else:
            return z
