from src.models.Generator import Generator
from src.models.Discriminator import Discriminator

import torch
import torch.optim as optim


class GAN():
    def __init__(self):
        super().__init__()

        self.latent_dim = 100

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        self.generator = Generator().to(self.device)
        self.discriminator = Discriminator().to(self.device)

        self.adversarial_loss = torch.nn.BCELoss().to(self.device)

        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def train(self, dataloader):
        num_epochs = 1

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader):

                real_spectrograms = data.to(self.device)
                real = torch.ones((real_spectrograms.size(0), 1), device=self.device)
                fake = torch.zeros((real_spectrograms.size(0), 1), device=self.device)

                # -----------------
                #  Train Generator
                # -----------------
                self.optimizer_G.zero_grad()
                z = torch.randn((real_spectrograms.size(0), self.latent_dim), device=self.device)

                gen_spectrograms = self.generator(z)
                g_loss = self.adversarial_loss(self.discriminator(gen_spectrograms), real)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                real_loss = self.adversarial_loss(self.discriminator(real_spectrograms), real)
                fake_loss = self.adversarial_loss(self.discriminator(gen_spectrograms.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(f"[Epoch {epoch}/{num_epochs}] [Batch {i}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
