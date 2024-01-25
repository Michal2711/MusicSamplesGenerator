import torch
import torch.optim as optim
import torch.nn.functional as F
import os

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
        
        if type(self.num_epochs_per_resolution) is list:
            self.num_epochs_per_resolution = self.num_epochs_per_resolution[-1]

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
    
    def optimize_discriminator(self, real_images, b_size, feature_vectors):

        self.optimizer_D.zero_grad()

        noise = self.create_noise(batch_size=b_size)

        if self.acgan:
            feature_vectors = feature_vectors.view(b_size, -1, 1, 1)
            noise = torch.cat((noise, feature_vectors), dim=1)

        fake_images = self.generator(noise)

        if self.acgan:
            fake_output, fake_features_vector = self.discriminator(fake_images.detach(), get_features=True)
            real_output, real_features_vector = self.discriminator(real_images, get_features=True)
        else:
            fake_output, fake_features_vector = self.discriminator(fake_images.detach(), get_features=False)
            real_output, real_features_vector = self.discriminator(real_images, get_features=False)
            
        real_output = real_output.view(-1)
        fake_output = fake_output.view(-1)

        if self.acgan:
            self.classification_penalty(disc_output=real_features_vector,
                                        target=feature_vectors,
                                        weight=10.0,
                                        backward=True)
            self.classification_penalty(disc_output=fake_features_vector,
                                        target=feature_vectors,
                                        weight=5.0,
                                        backward=True)

        d_loss = -(torch.mean(real_output) - torch.mean(fake_output)) # maximazing

        if self.epsilonD > 0:
            eps = (real_output[0] ** 2).sum() * self.epsilonD
            d_loss += eps 

        d_loss.backward(retain_graph=True)
        self.optimizer_D.step()
        
        # Clip weights of discriminator
        for p in self.discriminator.parameters():
            p.data.clamp_(-self.c, self.c)

        # if self.epsilonD > 0:
        #     loss_epsilon = (real_output[:, -1] ** 2).sum() * self.epsilonD
        #     d_loss += loss_epsilon

        # d_gradient_norm = self.compute_gradient_norm(self.discriminator.parameters())
        # self.writer_losses.add_scalar("Gradient_norms/Discriminator", d_gradient_norm, global_step=resolution*num_epochs + epoch)


        return d_loss

    def optimize_generator(self, b_size, feature_vectors):
        
        self.optimizer_G.zero_grad()
        self.optimizer_D.zero_grad()

        noise = self.create_noise(batch_size=b_size)

        if self.acgan:
            feature_vectors = feature_vectors.view(b_size, -1, 1, 1)
            noise = torch.cat((noise, feature_vectors), dim=1)

        fake_images = self.generator(noise)

        if self.acgan:
            fake_output, fake_feature_vectors = self.discriminator(fake_images, get_features=True)
        else:
            fake_output, fake_feature_vectors = self.discriminator(fake_images, get_features=False)

        fake_output = fake_output.view(-1)

        if self.acgan:
            self.classification_penalty(fake_feature_vectors,
                                        feature_vectors,
                                        weight=10.0,
                                        backward=True)
            # self.classification_penalty(fake_output,
            #                             feature_vectors,
            #                             weight=5.0,
            #                             backward=True)

        g_loss = -torch.mean(fake_output)
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # g_gradient_norm = self.compute_gradient_norm(self.generator.parameters())
        # self.writer_losses.add_scalar("Gradient_norms/Generator", g_gradient_norm, global_step=resolution*num_epochs + epoch)

        return g_loss

    def optimize_parameters(self, real_images, feature_vectors, feature_vectors_fake=None):
        
        b_size = real_images.size(0)
        real_images = real_images.to(self.device).float()

        if self.acgan:
            self.feature_vectors = feature_vectors.to(self.device)

        for _ in range(self.n_critic):
            d_loss = self.optimize_discriminator(
                real_images=real_images, 
                b_size=b_size,
                feature_vectors=feature_vectors
            )

        g_loss = self.optimize_generator(
            b_size=b_size,
            feature_vectors=feature_vectors
        )

        return d_loss, g_loss, b_size