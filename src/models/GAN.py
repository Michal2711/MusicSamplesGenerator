from src.models.Generator import Generator
from src.models.Discriminator import Discriminator
from src.models.Discriminator3 import Discriminator3
from src.models.Generator2 import Generator2
from src.models.Generator3 import Generator3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import librosa.display
import matplotlib.pyplot as plt

class GAN(nn.Module):
    def __init__(self,
                 height, 
                 width, 
                 num_epochs = 10, 
                 latent_dim=128,
                 output_dim=1,
                 lr=0.0002, 
                 loss='MSE'):
        super(GAN, self).__init__()

        self.num_epochs = num_epochs
        self.latent_dim = latent_dim
        self.height = height
        self.width = width
        self.lr = lr

        self.set_device()
        self.init_generator()
        self.init_discriminator()

        if loss not in ['MSE', 'BCE']:
            raise ValueError(
                "Loss type incorrect. Possibilities: ['MSE', 'BCE']"
            )
        elif loss == 'MSE':
            self.criterion = torch.nn.MSELoss().to(self.device)
        else:
            self.criterion = torch.nn.BCELoss().to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=2*self.lr, betas=(0.5, 0.999))
        self.scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.99)
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.scheulder_D = optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.99)

        noise = torch.randn(32, self.latent_dim, device=self.device)

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1        

    def init_generator(self):
        self.generator = Generator3(self.height, self.width, 'mel', self.latent_dim).to(self.device)

    def init_discriminator(self):
        self.discriminator = Discriminator3(self.height, self.width, 'mel').to(self.device)

    def train(self, dataloader):
        G_losses = []
        D_losses = []

        for epoch in range(self.num_epochs):
            D_fake_acc = []
            D_real_acc = []
            for i, data in enumerate(dataloader):

                self.optimizer_D.zero_grad()
                real_images = data.to(self.device)
                # print(f'real images shape: {real_images.size()}')
                b_size = real_images.size(0)
                label = torch.ones((b_size, ), dtype=torch.float, device=self.device)

                output = self.discriminator(real_images).view(-1)
                error_discriminator_real = self.criterion(output, label)
                error_discriminator_real.backward()
                D_real_acc.append(output.mean().item())

                # noise = torch.randn(b_size, self.latent_dim, device=self.device)
                noise = self.mu + self.sigma * torch.randn(b_size, self.latent_dim, device=self.device)

                fake_images = self.generator(noise)
                # print(f'fake images shape {fake_images.size()}')
                # label_fake = torch.zeros((b_size, ), dtype=torch.float, device=self.device)
                label_fake = torch.full((b_size,), 0, device=self.device )
                # print(f'fake images shape: {fake_images.shape}')
                output = self.discriminator(fake_images.detach()).view(-1)
                error_discriminator_fake = self.criterion(output, label_fake)
                error_discriminator_fake.backward()
                D_fake_acc.append(output.mean().item())

                error_discriminator = error_discriminator_real + error_discriminator_fake
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.01)
                self.optimizer_D.step()

                # update G Network
                self.optimizer_G.zero_grad()
                # label_real = torch.ones((b_size, ), dtype=torch.float, device=self.device)
                label_real = torch.full((b_size,), 1, device=self.device )
                output = self.discriminator(fake_images).view(-1)
                error_generator = self.criterion(output, label_real)
                error_generator.backward()
                # D_G_z2 = output.mean().item()
                self.optimizer_G.step()

                G_losses.append(error_generator.item())
                D_losses.append(error_discriminator.item())
                
            print(f"Epoch: {epoch}, discrimiantor fake error: {np.mean(D_fake_acc):.3}, discriminator real acc: {np.mean(D_real_acc):.3}")
            print(f"Epoch: {epoch} Generator loss: {np.mean(G_losses):.3}, Discriminator loss: {np.mean(D_losses):.3}")
            self.scheduler_G.step()
            self.scheulder_D.step()
            if epoch % 10 == 0:
                with torch.no_grad():
                    fake_spectrograms = self.generator(self.fixed_noise).detach().cpu()

                fig, axs = plt.subplots(8, 4, figsize=(150, 60))

                for i, spec in enumerate(fake_spectrograms):
                    row = i // 4
                    col = i % 4
                    spec = spec.squeeze(0)
                    img = librosa.display.specshow(spec.numpy(),
                                                   x_axis='time',
                                                   y_axis='log',
                                                   ax=axs[row, col])
                    axs[row, col].set_title(f'{i}', fontsize=20)
                    fig.colorbar(img, ax=axs[row, col], format=f'%0.2f')
                plt.show()
