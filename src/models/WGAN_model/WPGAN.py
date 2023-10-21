import torch
import torch.optim as optim
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter

from models.PGAN_model.PGAN import PGAN

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

    def train(self, dataloader, checkpoint_path=None):

        save_interval = 2
        if checkpoint_path is not None:
            epoch, resolution = self.load_checkpoint(checkpoint_path=checkpoint_path)
            print(f'Resuming training from epoch {epoch} at resolution {resolution}')
        else:
            epoch = 0
            resolution = 0

        for resolution in range(resolution, self.n_blocks):

            if type(self.num_epochs_per_resolution) is list:
                fade_epochs = int(self.num_epochs_per_resolution[resolution] * self.fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution[resolution]
            else:
                fade_epochs = int(self.num_epochs_per_resolution * self.fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution

            for epoch in range(epoch, num_epochs):
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

                        d_loss = -(torch.mean(real_output) - torch.mean(fake_output)) # maximazing
                        d_loss.backward()
                        self.writer_losses.add_scalar("Loss_discriminator/train", d_loss, global_step=resolution*num_epochs + epoch)
                        self.optimizer_D.step()

                        # Clip weights of discriminator
                        for p in self.discriminator.parameters():
                            p.data.clamp_(-self.c, self.c)

                    self.optimizer_G.zero_grad() 
                    fake_output = self.discriminator(fake_images).view(-1)
                    g_loss = -torch.mean(fake_output)
                    self.writer_losses.add_scalar("Loss_generator/train", g_loss, global_step=resolution*num_epochs + epoch)
                    g_loss.backward()
                    self.optimizer_G.step()

                print(f"Resolution {resolution} - Epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")

                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(model="WPGAN", resolution=resolution, epoch=epoch+1)

                with torch.no_grad():
                    test_noise = self.create_noise(batch_size=b_size)
                    fake_images = self.generator(test_noise)

                    real_tensor_grid = self.spectrograms_to_tensor_grid(real_images_low_res.cpu().numpy())
                    fake_tensor_grid = self.spectrograms_to_tensor_grid(fake_images.cpu().numpy())

                    self.writer_image_real.add_image("Real", real_tensor_grid, global_step=resolution*num_epochs + epoch)
                    self.writer_image_fake.add_image("Fake", fake_tensor_grid, global_step=resolution*num_epochs + epoch)

            epoch = 0
            if resolution < self.n_blocks-1:
                self.add_new_block(self.depths[resolution+1])

        self.add_hparams_to_writer(final_d_loss=d_loss.item(), final_g_loss=g_loss.item())
        spectrograms = next(iter(dataloader))
        spectrograms = spectrograms.to(self.device)
        self.add_models_to_writer(spectrograms)
        self.flush_all_writers()
   