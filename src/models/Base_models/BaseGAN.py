import torch
import torch.nn as nn
from abc import abstractmethod
import numpy as np
import librosa
from datetime import datetime
import os
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class BaseGAN(nn.Module):
    def __init__(self,
                 latent_dim=256,
                 output_dim=1,
                 lr=0.0002, 
                 batch_size = 32,
                 loss='MSE',
                 gpu=True):

        r"""
        Base class for Generative Adversarial Networks (GANs).

        This class provides a skeletal structure for GANs, with methods and attributes
        that are common across different types of GANs. It's designed to be subclassed,
        with subclasses implementing the abstract methods `get_generator`, `get_discriminator`,
        `get_optimizer_G`, `get_optimizer_D`, and `train`.

        Args:
            latent_dim (int): The size of the random noise vector used as input for the generator.
            output_dim (int): The size of the output vector of the discriminator.
            lr (float): The learning rate for the optimizers.
            loss (string): The loss function used in training. Must be one of ['MSE', 'BCE', 'WGAN'].
        """
        super(BaseGAN, self).__init__()

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.lr = lr
        self.batch_size = batch_size
        self.loss = loss
        self.gpu = gpu
        self.checkpoint_dir = ""
        self.model_save_dir = ""

        if loss not in ['MSE', 'BCE', 'WGAN']:
            raise ValueError(
                "Loss type incorrect. Possibilities: ['MSE', 'BCE', 'WGAN']"
            )

        self.set_device()

    def set_device(self):
        if self.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1        
    
    @abstractmethod
    def get_generator(self):
        pass

    @abstractmethod
    def get_discriminator(self):
        pass

    @abstractmethod
    def get_optimizer_G(self):
        pass

    @abstractmethod
    def get_optimizer_D(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    def create_noise(self, batch_size):
        noise = torch.randn(batch_size, self.latent_dim, 1, 1).to(self.device)
        return noise

    # def weights_init(self, m):
    #     classname = m.__class__.__name__
    #     if classname.find('Conv') != -1:
    #         print(f'classname: {classname}')
    #         nn.init.normal_(m.weight.data, 0.0, 0.02)
    #     elif classname.find('BatchNorm2d') != -1:
    #         nn.init.normal_(m.weight.data, 1.0, 0.02)
    #         nn.init.constant_(m.bias.data, 0)

    def update_solvers_device(self):
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        self.optimizer_D = self.get_optimizer_D()
        self.optimizer_G = self.get_optimizer_G()

        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

    def set_writers(self, base_log_dir):
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        base_log_dir = f'{base_log_dir}/{current_time}'

        self.writer_losses = SummaryWriter(log_dir=f"{base_log_dir}/losses")
        self.writer_image_real = SummaryWriter(log_dir=f"{base_log_dir}/images/real")
        self.writer_image_fake = SummaryWriter(log_dir=f"{base_log_dir}/images/fake")
        self.writer_hparams = SummaryWriter(log_dir=f"{base_log_dir}/hparams")
        self.writer_gen_model = SummaryWriter(log_dir=f"{base_log_dir}/models/gen")
        self.writer_disc_model = SummaryWriter(log_dir=f"{base_log_dir}/models/disc")

    def flush_all_writers(self):
        self.writer_losses.flush()
        self.writer_image_real.flush()
        self.writer_image_fake.flush()
        self.writer_hparams.flush()
        self.writer_gen_model.flush()
        self.writer_disc_model.flush()

    def close_all_writers(self):
        self.writer_losses.close()
        self.writer_image_real.close()
        self.writer_image_fake.close()
        self.writer_hparams.close()
        self.writer_gen_model.close()
        self.writer_disc_model.close()

    def spectrograms_to_tensor_grid(self, spectrograms):
        tensors = []
        
        for spec in spectrograms:
            spec = spec.squeeze()
            spec = np.abs(spec)
            power_to_db = librosa.power_to_db(spec, ref=np.max)
            fig = plt.figure(figsize=(8, 7))
            librosa.display.specshow(power_to_db, sr=22050, x_axis='time', y_axis='mel', cmap='magma', hop_length=512)
            plt.colorbar(label='dB')
            # plt.xlabel('Time', fontdict=dict(size=15))
            # plt.ylabel('Frequency', fontdict=dict(size=15))
            fig.canvas.draw()
            img_arr = np.array(fig.canvas.renderer.buffer_rgba())
            tensors.append(ToTensor()(img_arr))
            
            plt.close(fig)

        grid_tensor = make_grid(tensors, normalize=True) 
        return grid_tensor
    
    def create_dir_for_saving(self, current_dir, model):
        self.checkpoint_dir = os.path.join(current_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.model_save_dir = f"../../../models/saved_models/{model}/"
        self.model_save_dir = os.path.join(current_dir, self.model_save_dir)
        os.makedirs(self.model_save_dir, exist_ok=True)

    def save_checkpoint(self, model, resolution, epoch):
        checkpoint_name = f"{model}_checkpoint_res{resolution}_epoch{epoch}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'epoch': epoch,
            'resolution': resolution,
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

        epoch = checkpoint['epoch']
        resolution = checkpoint['resolution']

        return epoch, resolution

    def save_full_model(self, model_name):
        model_path = os.path.join(self.model_save_dir, model_name)

        torch.save({
            'generator': self.generator,
            'discriminator': self.discriminator,
            'optimizer_G_state_dict': self.get_optimizer_G().state_dict(),
            'optimizer_D_state_dict': self.get_optimizer_D().state_dict(),
        }, model_path)

        print(f'Full model saved: {model_path}')
