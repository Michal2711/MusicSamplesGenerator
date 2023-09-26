from src.models.Discriminator3 import Discriminator3
from src.models.Generator3 import Generator3

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

from abc import abstractmethod

class BaseGAN():
    def __init__(self,
                 latent_dim=128,
                 output_dim=1,
                 lr=0.0002, 
                 loss='MSE',
                 wgan_gp=0.0,
                 epsilonD=0.0,
                 g_activation=None,
                 **kwargs):

        r"""
        Args:
            latent_dim (int):
            output_dim (int):
            lr (float):
            loss (string):
            wgan_gp (float):
            epsilonD (float):
        """

        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.lr = lr
        self.wgan_gp = wgan_gp
        self.epsilonD = epsilonD
        self.g_activation = g_activation

        if loss not in ['MSE', 'BCE', 'WGANGP']:
            raise ValueError(
                "Loss type incorrect. Possibilities: ['MSE', 'BCE', 'WGANGP']"
            )


        self.set_device()
        self.generator = self.get_generator()
        self.discriminator = self.get_discriminator()

        self.update_solvers_device()

    def set_device(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.n_devices = torch.cuda.device_count()
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1        
    
    @abstractmethod
    def get_generator(self):
        r"""
            Define generator here
        """
        pass

    @abstractmethod
    def get_discriminator(self):
        r"""
            Define discriminator here
        """
        pass

    @abstractmethod
    def get_optimizer_G(self):
        r"""
            Define optimizer for generator here
        """
        pass

    @abstractmethod
    def get_optimizer_D(self):
        r"""
            Define optimizer for discriminator here
        """
        pass

    def get_original_generator(self):
        if isinstance(self.generator, nn.DataParallel):
            return self.generator.module
        return self.generator

    def get_original_discriminator(self):
        if isinstance(self.discriminator, nn.DataParallel):
            return self.discriminator.module
        return self.discriminator

    def get_model_parameters(self):

        state_generator = self.get_original_generator().state_dict()
        state_discriminator = self.get_original_discriminator().state_dict()

        state = {
            'config': self.config,
            'generator': state_generator,
            'discriminator': state_discriminator
        }
        return state

    def create_noise(self, n):
        noise = torch.randn(n, self.latent_dim).to(self.device)
        return noise

    def update_solvers_device(self):

        if not isinstance(self.discriminator, nn.DataParallel):
            self.discriminator = nn.DataParallel(self.discriminator)
        if not isinstance(self.generator, nn.DataParallel):
            self.generator = nn.DataParallel(self.generator)

        self.discriminator.to(self.device)
        self.generator.to(self.device)

        self.optimizer_D = self.get_optimizer_D()
        self.optimizer_G = self.get_optimizer_G()

        self.optimizer_D.zero_grad()
        self.optimizer_G.zero_grad()

    #TODO
    def optimize_parameters(self, input_batch):
        pass

    #TODO
    def load_state_dict(slef, in_state, load_g=True, load_d=True, load_config=True, finetuning=False):
        pass
    
    #TODO
    def classification_penalty(self, d_output, target, weight, backward=True):
        pass

    def save(self, path, saveTrainTmp=False):
        torch.save(self.get_model_parameters(saveTrainTmp=saveTrainTmp), path)

    def load(self, path="", in_state=None, load_g=True, load_d=True, load_config=True, finetuning=False):
        in_state = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")

        self.load_state_dict(in_state, load_g, load_d, load_config, finetuning)

    def count_params(self):

        generator = self.get_original_generator()
        discriminator = self.get_original_discriminator()
        param_count = dict(G=dict(total=0), D=dict(total=0))

        for name, params in generator.named_parameters():
            if params.requires_grad == True:
                n_params = params.numel()
                param_count['generator'][name]
                param_count['generator']['total'] += n_params

        for name, params in discriminator.named_parameters():
            if params.requires_grad == True:
                n_params = params.numel()
                param_count['discriminator'][name]
                param_count['discriminator']['total'] += n_params

        return json.dumps(param_count, indent=4)
