import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from torchvision.utils import make_grid

from models.Base_models.BaseGAN import BaseGAN
from models.PGAN_model.PDiscriminator import PDiscriminator
from models.PGAN_model.PGenerator import PGenerator
from models.ACGAN.ACGAN import ACGAN
from models.utils import init_seed

class PGAN(BaseGAN):
    def __init__(self,
                 depths,
                 init_resolution_size=(8, 4),
                 num_epochs_per_resolution = 10,
                 gen_output_dim = 1,
                 negative_slope=0.2,
                 fade_in_percentage=0.5,
                 save_interval=2,
                 normalization=True,
                 mini_batch_normalization=False,
                 epsilon_D = 0.001,
                 gen_type = 'audio',
                 feature_size=0,
                 features_keys_order = None,
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
        super().__init__(*args, **kwargs)

        self.init_depth = depths[0]
        self.depths = depths
        self.negative_slope = negative_slope
        self.fade_in_percentage = fade_in_percentage
        self.save_interval = save_interval
        self.normalization = normalization
        self.mini_batch_normalization = mini_batch_normalization
        self.gen_type = gen_type
        self.init_resolution_size = init_resolution_size
        self.num_epochs_per_resolution = num_epochs_per_resolution
        self.gen_output_dim = gen_output_dim
        self.n_blocks = len(depths)
        self.alpha = 0
        self.epsilonD = epsilon_D
        self.classification_criterion = None
        self.feature_size = feature_size
        self.features_keys_order = features_keys_order
        self.initialize_classification_criterion()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.create_dir_for_saving(current_dir=current_dir, model="PGAN")

        self.generator = self.get_generator().to(self.device)
        self.discriminator = self.get_discriminator().to(self.device)

        if self.loss != 'WGAN':
            self.optimizer_G = self.get_optimizer_G()
            self.optimizer_D = self.get_optimizer_D()

            self.optimizer_D.zero_grad()
            self.optimizer_G.zero_grad()
            
            self.set_writers("runs/PGAN")
            # self.criterion = nn.MSELoss()
            # self.criterion = nn.BCELoss()
            self.criterion = nn.BCEWithLogitsLoss()

    def get_generator(self):
        generator = PGenerator(
            init_depth=self.init_depth,
            init_resolution_size=self.init_resolution_size,
            latent_dim=self.latent_dim,
            LReLU_negative_slope=self.negative_slope,
            normalization=self.normalization,
            output_depth=self.gen_output_dim,
            feature_size=self.feature_size
        )
        return generator

    def get_discriminator(self):
        discriminator = PDiscriminator(
            last_depth=self.init_depth,
            init_resolution_size=self.init_resolution_size,
            LReLU_negative_slope=self.negative_slope,
            normalization=self.normalization,
            input_depth=self.gen_output_dim,
            feature_size=self.feature_size
        )
        return discriminator
    
    def get_optimizer_G(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
                    betas=[0, 0.99], lr=self.lr)

    def get_optimizer_D(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.discriminator.parameters()),
                          betas=[0, 0.99], lr=self.lr)

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

    # def add_models_to_writer(self, spectrograms):
    #     noise = self.create_noise(batch_size=self.batch_size)
    #     noise = noise.to(self.device)
    #     self.writer_gen_model.add_graph(self.generator, noise)
    #     # self.writer_disc_model.add_graph(self.discriminator, spectrograms)
    #     discriminator_for_tracing = lambda x: self.discriminator(x, for_tracing=True)
    #     self.writer_disc_model.add_graph(discriminator_for_tracing, spectrograms)

    def add_new_block(self, new_depth):
        if type(new_depth) is list:
            self.generator.add_next_block(new_depth[0])
            self.discriminator.add_next_block(new_depth[1])
        else:
            self.generator.add_next_block(new_depth)
            self.discriminator.add_next_block(new_depth)

        self.update_solvers_device()

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
        
    def compute_gradient_norm(self, parameters):
        """
        Compute the 2-norm of gradients for all trainable parameters.
        """
        total_norm = 0.0
        for param in parameters:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def initialize_classification_criterion(self):
        if self.acgan and self.features_keys_order is not None:
            self.classification_criterion = ACGAN(features_keys_order=self.features_keys_order)

    def classification_penalty(self, disc_output, target, weight, backward=True, skip_features=False):
        if self.classification_criterion is not None:
            loss = weight * \
                self.classification_criterion.get_criterion(disc_output, target, skip_features=skip_features)
            if torch.is_tensor(loss):
                if backward:
                    loss.backward(retain_graph=True)

    def optimize_generator(self, b_size, fake_labels):

        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        noise = self.create_noise(batch_size=b_size)

        fake_images = self.generator(noise)
        fake_pred, fake_features_pred = self.discriminator(fake_images, get_features=False)

        fake_pred = fake_pred.squeeze(1).squeeze(1).squeeze(1)

        g_loss = self.criterion(fake_pred, fake_labels)
        g_loss.backward()

        # g_gradient_norm = self.compute_gradient_norm(self.generator.parameters())
        # self.writer_losses.add_scalar("Gradient_norms/Generator", g_gradient_norm, global_step=resolution*num_epochs + epoch)

        self.optimizer_G.step()

        return g_loss
    
    def optimize_discriminator(self, real_images, b_size, real_labels, fake_labels):

        noise = self.create_noise(batch_size=b_size)
        fake_images = self.generator(noise).detach()

        self.optimizer_D.zero_grad()

        real_pred, real_features_pred = self.discriminator(real_images, get_features=False)
        fake_pred, fake_features_pred = self.discriminator(fake_images, get_features=False)

        real_pred = real_pred.squeeze(1).squeeze(1).squeeze(1)
        fake_pred = fake_pred.squeeze(1).squeeze(1).squeeze(1)

        d_loss_real = self.criterion(real_pred, real_labels)
        d_loss_fake = self.criterion(fake_pred, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()

        # d_gradient_norm = self.compute_gradient_norm(self.discriminator.parameters())
        # self.writer_losses.add_scalar("Gradient_norms/Discriminator", d_gradient_norm, global_step=resolution*num_epochs + epoch)

        self.optimizer_D.step()

        return d_loss

    def optimize_parameters(self, real_images, feature_vectors, feature_vectors_fake=None):
        b_size = real_images.size(0)
        real_images = real_images.to(self.device).float()
        real_labels = torch.ones((b_size,), dtype=torch.float, device=self.device)
        fake_labels = torch.zeros((b_size,), dtype=torch.float, device=self.device)


        d_loss = self.optimize_discriminator(
            real_images=real_images,
            b_size=b_size,
            real_labels=real_labels,
            fake_labels=fake_labels, 
        )

        g_loss = self.optimize_generator(
            b_size=b_size,
            fake_labels=fake_labels,
        )
        
        return d_loss, g_loss, b_size

    def trainOnEpoch(self, dataloader, resolution, epoch, num_epochs, fade_epochs):
        if resolution > 0:
            if epoch < fade_epochs:
                self.update_alpha((epoch+1)/ fade_epochs)
            elif epoch == fade_epochs:
                self.update_alpha(1)

        for _, (data, feature_vectors) in enumerate(dataloader):
            resolution_size = self.generator.get_output_size()
            real_images_low_res = F.interpolate(data, size=resolution_size, mode="nearest")

            d_loss, g_loss, b_size= self.optimize_parameters(real_images=real_images_low_res, feature_vectors=feature_vectors, feature_vectors_fake=None)

        return d_loss, g_loss, b_size, real_images_low_res
        
    def train(self, dataloader, checkpoint_path=None):

        init_seed()

        self.generator.train()
        self.discriminator.train()

        if checkpoint_path is not None:
            epoch, resolution = self.load_checkpoint(checkpoint_path=checkpoint_path)
            print(f'Resuming training from epoch {epoch} at resolution {resolution}')
        else:
            epoch = 0
            resolution = 0
            
        global_step = 0

        for resolution in range(resolution, self.n_blocks):
            if type(self.num_epochs_per_resolution) is list:
                fade_epochs = int(self.num_epochs_per_resolution[resolution] * self.fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution[resolution]
            else:
                fade_epochs = int(self.num_epochs_per_resolution * self.fade_in_percentage)
                num_epochs = self.num_epochs_per_resolution

            for epoch in range(epoch, num_epochs):
                d_loss, g_loss, b_size, real_images_low_res = self.trainOnEpoch(dataloader=dataloader, resolution=resolution, epoch=epoch, num_epochs=num_epochs, fade_epochs=fade_epochs)

                print(f"Resolution {resolution} - Epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")
                global_step += 1
                # self.writer_losses.add_scalar("Losses/Discriminator", d_loss.item(), global_step=resolution*num_epochs + epoch)
                # self.writer_losses.add_scalar("Losses/Generator", g_loss.item(), global_step=resolution*num_epochs + epoch)
                
                # if resolution > 4 and abs(g_loss) < 25.0:
                #     self.save_checkpoint(model='GIT', resolution=resolution, epoch=epoch+1)

                # if (epoch + 1) % self.save_interval == 0 and resolution > 4:
                #     self.save_checkpoint(model="PGAN", resolution=resolution, epoch=epoch+1)

                with torch.no_grad():
                    test_noise = self.create_noise(batch_size=b_size)
                    if self.acgan:
                        random_feature_vectors = self.classification_criterion.create_random_feature_vectors(b_size=b_size)
                        random_feature_vectors = random_feature_vectors.view(b_size, -1, 1, 1).to(self.device)
                        test_noise = torch.cat((test_noise, random_feature_vectors), dim=1)

                    fake_images = self.generator(test_noise)

                    if self.gen_type == 'audio':
                        real_tensor_grid = self.spectrograms_to_tensor_grid(real_images_low_res.cpu().numpy())
                        fake_tensor_grid = self.spectrograms_to_tensor_grid(fake_images.cpu().numpy())
                    elif self.gen_type == 'pictures':    
                        real_tensor_grid = make_grid(real_images_low_res.cpu(), normalize=True)
                        fake_tensor_grid = make_grid(fake_images.cpu(), normalize=True)

                    self.writer_image_real.add_image("Real", real_tensor_grid, global_step=global_step)
                    self.writer_image_fake.add_image("Fake", fake_tensor_grid, global_step=global_step)

            epoch = 0
            if resolution < self.n_blocks-1:
                self.add_new_block(self.depths[resolution+1])

        self.add_hparams_to_writer(final_d_loss=d_loss.item(), final_g_loss=g_loss.item())
        # spectrograms, feature_vector = next(iter(dataloader))
        # spectrograms = spectrograms.to(self.device)
        # print(type(spectrograms))
        # self.add_models_to_writer(spectrograms)
        self.flush_all_writers()