import torch
import torch.optim as optim
import torch.nn.functional as F
import tqdm

from models.PGAN_model.PGAN import PGAN
# from models.utils import gradient_penalty

class WPGAN_GP(PGAN):
    def __init__(self, n_critic, lambda_gp, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.n_critic = n_critic
        self.lambda_gp = lambda_gp
        self.optimizer_G = self.get_optimizer_G()
        self.optimizer_D = self.get_optimizer_D()

        self.set_writers("runs/WPGAN-GP")

    def get_optimizer_G(self):
        return optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.0, 0.99))
    
    def get_optimizer_D(self):
        return optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.0, 0.99))
    
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

    def train_fc(self, dataloader):
        
        scaler_disc = torch.cuda.amp.GradScaler()
        scaler_gen = torch.cuda.amp.GradScaler()

    def train(self, dataloader):

        scaler_disc = torch.cuda.amp.GradScaler()
        scaler_gen = torch.cuda.amp.GradScaler()

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
                    
                loop = tqdm(dataloader, leave=True)
                for batch_idx, (real, _) in enumerate(loop):
                    real = real.to(self.device)
                    cur_batch_size = real.shape[0]

                    noise = self.create_noise(cur_batch_size)

                    with torch.cuda.amp.autocast():
                        fake = self.generator(noise)
                        discriminator_real = self.discriminator(real)
                        discriminator_fake = self.discriminator(fake.detach())

                        gp = gradient_penalty(self.discriminator, real, fake, device=self.device)
                        loss_disc = (
                            -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                            + 10 * gp
                            + (0.001 * torch.mean(discriminator_real ** 2))
                        )
                    
                    self.optimizer_D.zero_grad()
                    scaler_disc.scale(loss_disc).backward()
                    scaler_disc.step(self.optimize_discriminator)
                    scaler_disc.update()

                    with torch.cuda.amp.autocast():
                        gen_fake = self.discriminator(fake)
                        loss_gen = -torch.mean(gen_fake)

                    self.optimizer_G.zero_grad()
                    scaler_gen.scale(loss_gen).backward()
                    scaler_gen.step(self.optimizer_G)
                    scaler_gen.update()

                print(f"Resolution {resolution} - Epoch {epoch+1}/{num_epochs} - D Loss: {d_loss.item()} - G Loss: {g_loss.item()}")

            if resolution < self.n_blocks-1:
                self.add_new_block(self.depths[resolution+1])
        
        self.add_hparams_to_writer(final_d_loss=d_loss.item(), final_g_loss=g_loss.item())
        spectrograms = next(iter(dataloader))
        spectrograms = spectrograms.to(self.device)
        self.add_models_to_writer(spectrograms)
        self.flush_all_writers()

    