import torch.optim as optim

from Base_models.BaseGAN import BaseGAN
from PGenerator import PGenerator
from PDiscriminator import PDiscriminator

class PGAN(BaseGAN):

    def __init__(self,
                 depths,
                 latent_dim=100,
                 negative_slope=0.2,
                 normalization=True,
                 mini_batch_normalization=False,
                 init_resolution_size=(8, 5)):
        r"""
        Args:

        """

        self.init_depth = depths[0]
        self.latent_dim = latent_dim
        self.negative_slope = negative_slope
        self.normalization = normalization
        self.mini_batch_normalization = mini_batch_normalization
        self.init_resolution_size = init_resolution_size
        self.n_blocks = len(depths)
        self.alpha = 0

        BaseGAN.__init__(self, latent_dim=latent_dim)

    # TODO PGAN + BaseGAN
    def update_config(self, config):
        pass

    def get_generator(self):
        generator = PGenerator(
            init_depth=self.init_depth,
            init_resolution_size=self.init_resolution_size,
            latent_dim=self.latent_dim,
            LReLU_negative_slope=self.negative_slope,
            normalization=self.normalization
        )

        return generator

    def get_discriminator(self):
        discriminator = PDiscriminator(
            init_depth=self.init_depth,
            init_resolution_size=self.init_resolution_size,
            LReLU_negative_slope=self.negative_slope,
            normalization=self.normalization
        )

        return discriminator
    
    def get_optimizer_G(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                          betas=[0, 0.99], lr=self.lr)
    
    def get_optimizer_D(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()),
                          betas=[0, 0.99], lr=self.lr)
    
    def add_scale(self, new_depth):

        self.generator = self.get_original_generator()
        self.discriminator = self.get_original_discriminator()

        if type(new_depth) is list:
            self.generator.addScale(new_depth[0])
            self.discriminator.addScale(new_depth[1])
        else:
            self.generator.addScale(new_depth)
            self.discriminator.addScale(new_depth)

        # TODO moze depthOtherScales sie przyda
        self.update_solvers_device()

    def update_alpha(self, new_alpha):
        self.get_original_generator().set_alpha(new_alpha)
        self.get_original_discriminator().set_alpha(new_alpha)

        self.alpha = new_alpha

    def get_output_size(self):
        return self.get_original_generator().getOutputSize()
    
    def test_generator(self, input):
        if type(input) == list:
            input = [i.to(self.device) for i in input]
        else:
            input = input.to(self.device)

        out = self.generator(input)
        if type(out) == list:
            return [o.detach().cpu() for o in out]
        else:
            return out.detach().cpu()
        
    def optimize_generator(self, losses):
        pass
        # batch_size = self.real_input.size(0)

        # self.optimizer_G.zero_grad()
        # self.optimizer_D.zero_grad()

        # input_vector = self.create_noise(batch_size)

        # fake_generation = self.generator(input_vector)
        # prediction_fake, _ = self.discriminator(fake_generation, True)

        # loss_generator_fake = self.loss.

    def optimize_discriminator(self, losses):
        pass

    def optimize_parameters(self, input_batch):
        pass
