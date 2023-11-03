import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

from models.Base_models.BaseGAN import BaseGAN
from models.PGAN_model.old_version.Pdisc_old import PDiscriminator
from models.PGAN_model.old_version.Pgen_old import PGenerator

class PGAN(BaseGAN):
    def __init__(self,
                 depths,
                 init_resolution_size=(8, 5),
                 num_epochs_per_resolution = 10,
                 gen_output_dim = 1,
                 negative_slope=0.2,
                 fade_in_percentage=0.5,
                 save_interval=2,
                 normalization=True,
                 mini_batch_normalization=False,
                 *args, **kwargs):
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
        super(PGAN, self).__init__(*args, **kwargs)

        self.init_depth = depths[0]
        self.depths = depths
        self.negative_slope = negative_slope
        self.fade_in_percentage = fade_in_percentage
        self.save_interval = save_interval
        self.normalization = normalization
        self.mini_batch_normalization = mini_batch_normalization
        self.init_resolution_size = init_resolution_size
        self.num_epochs_per_resolution = num_epochs_per_resolution
        self.gen_output_dim = gen_output_dim
        self.n_blocks = len(depths)
        self.alpha = 0

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.create_dir_for_saving(current_dir=current_dir, model="PGAN")

        self.generator = self.get_generator().to(self.device)
        self.generator.apply(self.weights_init)

        self.discriminator = self.get_discriminator().to(self.device)
        self.discriminator.apply(self.weights_init)

        if self.loss != 'WGAN':
            self.optimizer_G = self.get_optimizer_G()
            self.optimizer_D = self.get_optimizer_D()
            
            self.set_writers("runs/PGAN")
            self.criterion = nn.MSELoss()
            # self.criterion = nn.BCELoss()

    def get_generator(self):
        generator = PGenerator(
            init_depth=self.init_depth,
            init_resolution_size=self.init_resolution_size,
            latent_dim=self.latent_dim,
            LReLU_negative_slope=self.negative_slope,
            normalization=self.normalization,
            output_depth=self.gen_output_dim,
        )
        return generator

    def get_discriminator(self):
        discriminator = PDiscriminator(
            init_depth=self.init_depth,
            init_resolution_size=self.init_resolution_size,
            LReLU_negative_slope=self.negative_slope,
            normalization=self.normalization,
            input_depth=self.gen_output_dim,
        )
        return discriminator
    
    def get_optimizer_G(self):
        return optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
    
    def get_optimizer_D(self):
        return optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))

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
        }

        metrics = {
            'final_d_loss': final_d_loss,
            'final_g_loss': final_g_loss
        }
        
        self.writer_hparams.add_hparams(hparams, metrics)

    def add_models_to_writer(self, spectrograms):
        noise = self.create_noise(batch_size=self.batch_size)
        noise = noise.to(self.device)
        self.writer_gen_model.add_graph(self.generator, noise)
        self.writer_disc_model.add_graph(self.discriminator, spectrograms)

    def add_new_block(self, new_depth):
        if type(new_depth) is list:
            self.generator.add_next_block(new_depth[0])
            self.discriminator.add_next_block(new_depth[1])
        else:
            self.generator.add_next_block(new_depth)
            self.discriminator.add_next_block(new_depth)

        new_layers_gen = list(self.generator.children())[-1]
        new_layers_disc = list(self.discriminator.children())[-1]
        new_layers_gen.apply(self.weights_init)
        new_layers_disc.apply(self.weights_init)

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
        
    def train(self, dataloader, checkpoint_path=None):

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
                    self.writer_losses.add_scalar("Loss_discriminator/train", d_loss, global_step=resolution*num_epochs + epoch)
                    self.optimizer_D.step()

                    # 2. Train Generator
                    self.optimizer_G.zero_grad()
                    label = torch.ones((b_size,), dtype=torch.float, device=self.device)
                    output = self.discriminator(fake_images).view(-1)
                    g_loss = self.criterion(output, label)
                    self.writer_losses.add_scalar("Loss_generator/train", g_loss, global_step=resolution*num_epochs + epoch)
                    g_loss.backward()
                    self.optimizer_G.step()

                print(f"Resolution {resolution} - Epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")
                
                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(model="PGAN", resolution=resolution, epoch=epoch+1)

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
