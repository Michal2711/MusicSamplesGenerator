import torch
import torch.nn as nn
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter

class BaseGAN(nn.Module):
    def __init__(self,
                 latent_dim=100,
                 output_dim=1,
                 lr=0.0002, 
                 loss='MSE'):

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
        self.writer = SummaryWriter()

        if loss not in ['MSE', 'BCE', 'WGAN']:
            raise ValueError(
                "Loss type incorrect. Possibilities: ['MSE', 'BCE', 'WGAN']"
            )

        self.set_device()

    def set_device(self):
        if torch.cuda.is_available():
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

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator
    
    def create_noise(self, batch_size):
        noise = torch.randn(batch_size, self.latent_dim).to(self.device)
        return noise
    
    def close_writer(self):
        self.writer.close()

    def get_model_parameters(self):

        generator_params = self.generator.state_dict()
        discriminator_params = self.discriminator.state_dict()

        model_params = {
            'generator': generator_params,
            'discriminator': discriminator_params
        }
        return model_params

    def save(self, path):
        torch.save(self.get_model_parameters(), path)

    def load(self, path):
        state = torch.load(path, map_location=self.device)

        self.generator.load_state_dict(state['generator'])
        self.discriminator.load_state_dict(state['discriminator'])