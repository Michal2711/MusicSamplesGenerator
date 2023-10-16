import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models.Base_models.BaseGAN import BaseGAN
from models.PGAN_model.PDiscriminator import PDiscriminator
from models.PGAN_model.PGenerator import PGenerator

class PGAN(BaseGAN):

    def __init__(self,
                 depths,
                 latent_dim=100,
                 negative_slope=0.2,
                 normalization=True,
                 mini_batch_normalization=False,
                 init_resolution_size=(8, 5),
                 num_epochs_per_resolution = 10):
        r"""
        Progressive Growing GAN (PGAN) implementation.

        This class represents a GAN that starts training on low-resolution images and progressively 
        increases the resolution by adding new layers to the generator and discriminator networks. 
        This method helps in stabilizing the training process and allows the generation of high-quality images.

        Args:
            depths ([int]): A list of depths that represent the number of channels in each resolution block. 
                            Each depth corresponds to a specific stage of the 
                            progressively growing network.
            negative_slope (float): The negative slope parameter for the LeakyReLU activation function.
            normalization (bool): If set to True, normalization is applied within the networks. 
            mini_batch_normalization (bool): If set to True, mini-batch normalization is applied 
                                         within the networks.
            init_resolution_size ((int, int)): The initial resolution size at the beginning of the training.
            num_epochs_per_resolution (int): The number of training epochs for each resolution stage. 
            n_blocks (int): The number of blocks in the networks, derived from the length of `depths`.

        """
        super(PGAN, self).__init__(latent_dim=latent_dim)

        self.init_depth = depths[0]
        self.depths = depths
        self.negative_slope = negative_slope
        self.normalization = normalization
        self.mini_batch_normalization = mini_batch_normalization
        self.init_resolution_size = init_resolution_size
        self.num_epochs_per_resolution = num_epochs_per_resolution
        self.n_blocks = len(depths)
        self.alpha = 0

        self.generator = self.get_generator().to(self.device)
        self.discriminator = self.get_discriminator().to(self.device)
        self.optimizer_G = self.get_optimizer_G()
        self.optimizer_D = self.get_optimizer_D()

        self.criterion = nn.MSELoss()

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
        return optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0, 0.99))
    
    def get_optimizer_D(self):
        return optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0, 0.99))
    
    def add_new_block(self, new_depth):
        if type(new_depth) is list:
            self.generator.add_next_block(new_depth[0])
            self.discriminator.add_next_block(new_depth[1])
        else:
            self.generator.add_next_block(new_depth)
            self.discriminator.add_next_block(new_depth)

    def update_alpha(self, new_alpha):
        self.generator.set_alpha(new_alpha)
        self.discriminator.set_alpha(new_alpha)

        self.alpha = new_alpha
    
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
        
    def train(self, dataloader, fade_in_percentage=0.5):
        for resolution in range(self.n_blocks):

            if type(self.num_epochs_per_resolution) is list:
                fade_epochs = int(self.num_epochs_per_resolution[resolution] * fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution[resolution]
            else:
                fade_epochs = int(self.num_epochs_per_resolution * fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution

            for epoch in range(num_epochs):
                if resolution > 0:
                    if epoch < fade_epochs:
                        self.update_alpha((epoch+1)/ fade_epochs)
                    elif epoch == fade_epochs:
                        self.update_alpha(1)
                    
                for _, data in enumerate(dataloader):
                    # 1. Train Discriminator
                    self.optimizer_D.zero_grad()

                    real_images = data.to(self.device)
                    b_size = real_images.size(0)
                    label = torch.ones((b_size,), dtype=torch.float, device=self.device)
                    resolution_size = self.generator.get_output_size()
                    # real_images = F.interpolate(data, size=resolution_size, mode="nearest")
                    real_images_low_res = F.adaptive_avg_pool2d(real_images, output_size=resolution_size)
                    output = self.discriminator(real_images_low_res).view(-1)
                    real_loss = self.criterion(output, label)
                    real_loss.backward()

                    # fake images
                    noise = self.create_noise(batch_size=b_size)
                    fake_images = self.generator(noise)
                    label_fake = torch.zeros((b_size,), dtype=torch.float, device=self.device)
                    output = self.discriminator(fake_images.detach()).view(-1)
                    fake_loss = self.criterion(output, label_fake)
                    fake_loss.backward()

                    d_loss = (real_loss + fake_loss)
                    self.writer.add_scalar("Loss_discriminator/train", d_loss, global_step=resolution*num_epochs + epoch)
                    self.optimizer_D.step()

                    # 2. Train Generator
                    self.optimizer_G.zero_grad()
                    label = torch.ones((b_size,), dtype=torch.float, device=self.device)
                    output = self.discriminator(fake_images).view(-1)
                    g_loss = self.criterion(output, label)
                    self.writer.add_scalar("Loss_generator/train", g_loss, global_step=resolution*num_epochs + epoch)
                    g_loss.backward()
                    self.optimizer_G.step()

                print(f"Resolution {resolution} - Epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")

            if resolution < self.n_blocks-1:
                self.add_new_block(self.depths[resolution+1])
        self.writer.flush()
