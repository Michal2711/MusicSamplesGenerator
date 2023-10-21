import torch
import torch.optim as optim
import torch.nn.functional as F

from models.PGAN_model.PGAN import PGAN
from models.utils import gradient_penalty

class WGAN(PGAN):
    def __init__(self, n_critic, lambda_gp, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.optimizer_G = self.get_optimizer_G()
        self.optimizer_D = self.get_optimizer_D()

        self.set_writers("runs/WPGAN-GP")

    def get_optimizer_G(self):
        return optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.9))
    
    def get_optimizer_D(self):
        return optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.0, 0.9))
    
    def add_hparams_to_writer(self, final_d_loss = None, final_g_loss = None):
        hparams = {
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'learning_rate': self.lr,
            'batch_size': self.batch_size,
            'loss': self.loss,
            'depths': f"{self.depths}",
            'init_resolution_size': f"{self.init_resolution_size}",
            'num_epochs': len(self.depths) * self.num_epochs_per_resolution,
            'num_epochs_per_resolution': self.num_epochs_per_resolution,
            'negative_slope': self.negative_slope,
            'fade_in': self.fade_in_percentage,
            'normalization': self.normalization,
            'mini_batch_normalization': self.mini_batch_normalization,
            'lambda_gp': self.lambda_gp,
            'n_critic': self.n_critic
        }

        metrics = {
            'final_d_loss': final_d_loss,
            'final_g_loss': final_g_loss
        }
        
        self.writer_hparams.add_hparams(hparams, metrics)

    def train(self, dataloader):
        for resolution in range(self.n_blocks):

            if type(self.num_epochs_per_resolution) is list:
                fade_epochs = int(self.num_epochs_per_resolution[resolution] * self.fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution[resolution]
            else:
                fade_epochs = int(self.num_epochs_per_resolution * self.fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution

            for epoch in range(num_epochs):
                if resolution > 0:
                    if epoch < fade_epochs:
                        self.update_alpha((epoch+1)/ fade_epochs)
                    elif epoch == fade_epochs:
                        self.update_alpha(1)
                    
                for _, data in enumerate(dataloader):
                    
                    # 1. Train Discriminator
                    for _ in range(self.n_critic):

                        self.optimizer_D.zero_grad()

                        real_images = data.to(self.device)
                        b_size = real_images.size(0)
                        resolution_size = self.generator.get_output_size()
                        # real_images = F.interpolate(data, size=resolution_size, mode="nearest")
                        real_images_low_res = F.adaptive_avg_pool2d(real_images, output_size=resolution_size)
                        real_output = self.discriminator(real_images_low_res).view(-1)

                        # fake images
                        noise = self.create_noise(batch_size=b_size)
                        fake_images = self.generator(noise)
                        fake_output = self.discriminator(fake_images.detach()).view(-1)
                        gp = gradient_penalty(critic=self.discriminator, real=real_images, fake=fake_images, device=self.device)
                        d_loss = -(torch.mean(real_output) - torch.mean(fake_output)) + self.lambda_gp*gp # maximazing
                        d_loss.backward()
                        self.writer.add_scalar("Loss_discriminator/train", d_loss, global_step=resolution*num_epochs + epoch)
                        self.optimizer_D.step()

                    self.optimizer_G.zero_grad() 
                    fake_output = self.discriminator(fake_images).view(-1)
                    g_loss = -torch.mean(fake_output)
                    self.writer.add_scalar("Loss_generator/train", g_loss, global_step=resolution*num_epochs + epoch)
                    g_loss.backward()
                    self.optimizer_G.step()

                print(f"Resolution {resolution} - Epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")

            if resolution < self.n_blocks-1:
                self.add_new_block(self.depths[resolution+1])
        
        self.add_hparams_to_writer(final_d_loss=d_loss.item(), final_g_loss=g_loss.item())
        spectrograms = next(iter(dataloader))
        spectrograms = spectrograms.to(self.device)
        self.add_models_to_writer(spectrograms)
        self.flush_all_writers()

    