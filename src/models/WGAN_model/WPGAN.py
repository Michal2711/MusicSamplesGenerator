import torch
import torch.optim as optim
import torch.nn.functional as F
import os

from models.PGAN_model.PGAN import PGAN
from models.utils import finiteCheck

class WPGAN(PGAN):
    def __init__(self, c, n_critic, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.c = c
        self.n_critic = n_critic
        self.optimizer_G = self.get_optimizer_G()
        self.optimizer_D = self.get_optimizer_D()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.create_dir_for_saving(current_dir=current_dir, model="WPGAN")

        self.set_writers("runs/WPGAN")

    def get_optimizer_G(self):
        return optim.RMSprop(self.generator.parameters(), lr=self.lr)
    
    def get_optimizer_D(self):
        return optim.RMSprop(self.discriminator.parameters(), lr=self.lr)
    
    def add_hparams_to_writer(self, final_d_loss = None, final_g_loss = None):
        hparams = {
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.lr,
            'batch_size': self.batch_size,
            'loss': self.loss,
            'gpu': self.gpu,
            'depths': f"{self.depths}",
            'init_resolution_size': f"{self.init_resolution_size}",
            'num_epochs': len(self.depths) * self.num_epochs_per_resolution,
            'num_epochs_per_resolution': self.num_epochs_per_resolution,
            'negative_slope': self.negative_slope,
            'fade_in': self.fade_in_percentage,
            'normalization': self.normalization,
            'mini_batch_normalization': self.mini_batch_normalization,
            'c': self.c,
            'n_critic': self.n_critic
        }

        metrics = {
            'final_d_loss': final_d_loss,
            'final_g_loss': final_g_loss
        }
        
        self.writer_hparams.add_hparams(hparams, metrics)

    def optimize_generator(self, b_size, resolution, num_epochs, epoch):
        self.optimizer_G.zero_grad() 

        noise = self.create_noise(batch_size=b_size)
        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images).view(-1)
        g_loss = -torch.mean(fake_output)
        g_loss.backward()

        g_gradient_norm = self.compute_gradient_norm(self.generator.parameters())
        self.writer_losses.add_scalar("Gradient_norms/Generator", g_gradient_norm, global_step=resolution*num_epochs + epoch)

        finiteCheck(self.generator.parameters())
        self.optimizer_G.step()

        return g_loss
    
    def optimize_discriminator(self, real_images, b_size, resolution, num_epochs, epoch):
        self.optimizer_D.zero_grad()

        real_images = real_images.to(self.device)
        b_size = real_images.size(0)
        resolution_size = self.generator.get_output_size()
        real_images_low_res = F.interpolate(real_images, size=resolution_size, mode="nearest")
        # real_images_low_res = F.adaptive_avg_pool2d(real_images, output_size=resolution_size)
        real_output = self.discriminator(real_images_low_res).view(-1)

        # fake images
        noise = self.create_noise(batch_size=b_size)
        fake_images = self.generator(noise)
        fake_output = self.discriminator(fake_images.detach()).view(-1)

        d_loss = -(torch.mean(real_output) - torch.mean(fake_output)) # maximazing
        d_loss.backward()

        d_gradient_norm = self.compute_gradient_norm(self.discriminator.parameters())
        self.writer_losses.add_scalar("Gradient_norms/Discriminator", d_gradient_norm, global_step=resolution*num_epochs + epoch)

        finiteCheck(self.discriminator.parameters()) # check if needed
        self.optimizer_D.step()

        # Clip weights of discriminator
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.c, self.c)
        
        return d_loss

    def optimizeParameters(self, real_images, resolution, num_epochs, epoch):
        b_size = real_images.size(0)
        real_images = real_images.to(self.device).float()

        for _ in range(self.n_critic):
            d_loss = self.optimize_discriminator(
                real_images=real_images, 
                b_size=b_size,
                resolution=resolution,
                num_epochs=num_epochs,
                epoch=epoch
            )
        g_loss = self.optimize_generator(
            b_size=b_size,
            resolution=resolution,
            num_epochs=num_epochs,
            epoch=epoch
        )

        return d_loss, g_loss, b_size

    