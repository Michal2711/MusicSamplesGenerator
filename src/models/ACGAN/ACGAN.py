from random import randint
import random
import torch
import torch.nn.functional as F
import numpy as np


# ACGAN implementation based on DrumGAN model
class ACGAN:
    def __init__(self, features_keys_order, skip_features_fake=[]) -> None:
        
        self.features_keys_order = features_keys_order
        self.skip_features_fake = skip_features_fake
        self.n_features = len(features_keys_order)
        self.key_order = list(features_keys_order.keys())

        self.labels_order = {}
        self.feature_size = []
        self.att_loss = []

        for i, att in enumerate(self.features_keys_order):
            self.feature_size.append(len(self.features_keys_order[att]["values"]))
            self.labels_order[att] = {index: label for label, index in
                                enumerate(self.features_keys_order[att]["values"])}
            self.att_loss.append('cross-entropy')    

        self.label_weights = torch.tensor(
            [1.0 for x in range(sum(self.feature_size))])

        for i, key in enumerate(self.features_keys_order):
            if self.features_keys_order[key].get('weights', None) is not None:
                shift = sum(self.feature_size[:i])
                for value, weight in self.features_keys_order[key]['weights'].items():
                    self.label_weights[shift +
                                      self.labels_order[key][value]] = weight

    def create_random_feature_vectors(self, b_size):
        input_latent = []

        for i in range(self.n_features):
            C = self.feature_size[i]
            
            if self.key_order[i] == 'qualities':
                w = np.zeros((b_size, C), dtype='float32')
                for j in range(b_size):
                    num_qualities = np.random.randint(1, 6)
                    qualities = np.random.choice(C, num_qualities, replace=False)

                    if 0 in qualities and 1 in qualities:
                        qualities = list(filter(lambda x: x != random.choice([0, 1]), qualities))
                    if 3 in qualities and 4 in qualities:
                        qualities = list(filter(lambda x: x != random.choice([3, 4]), qualities))

                    w[j, qualities] = 1
                y = torch.tensor(w).view(b_size, C)
            else:
                v = np.random.randint(0, C, b_size)
                w = np.zeros((b_size, C), dtype='float32')
                w[np.arange(b_size), v] = 1
                y = torch.tensor(w).view(b_size, C)

            input_latent.append(y)

        return torch.cat(input_latent, dim=1)

    def get_criterion(self, output_D, target, skip_features=False):
        r"""
        Compute the conditional loss between the network's output and the
        target. This loss, L, is the sum of the losses Lc of the categories
        defined in the criterion. We have:

                 | Cross entropy loss for the class c if c is attached to a
                   classification task.
            Lc = | Multi label soft margin loss for the class c if c is
                   attached to a tagging task
        """
        loss = 0
        shift_input = 0
        shift_target = 0
        self.label_weights = self.label_weights.to(output_D.device)
        losses = []

        for i in range(self.n_features):
            C = self.feature_size[i]
            if self.key_order[i] not in self.skip_features_fake or not skip_features:

                loc_input = output_D[:, shift_input:(shift_input+C)]

                if self.att_loss[i] == 'mse':
                    loc_target = target[:, shift_target:shift_target + C]
                    loc_input = torch.sigmoid(loc_input)
                    loc_target = loc_target.reshape(loc_input.size())
                    loc_loss = F.mse_loss(loc_input, loc_target)
                    shift_target += C
                elif self.att_loss[i] == 'bce':
                    loc_target = target[:, shift_target:shift_target + C]
                    loc_input = torch.sigmoid(loc_input)
                    loc_input = loc_input.float()
                    loc_target = loc_target.float()
                    loc_loss = F.binary_cross_entropy(loc_input, loc_target)
                    shift_target += C
                else:
                    loc_target = target[:, shift_target]
                    loc_loss = F.cross_entropy(loc_input, loc_target.long(), 
                                          weight=self.label_weights[shift_input:(shift_input+C)])
                    shift_target += 1
                loss += loc_loss
                losses.append((loc_loss, self.key_order[i]))
            
            shift_input += C
        return loss
