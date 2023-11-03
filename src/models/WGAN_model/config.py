# # hyperparameters

# Basic model parameters
latent_dim = 100
output_dim=1
gen_output_dim=1
learning_rate = 5e-5
loss = 'WGAN'
batch_size = 64
gpu = True

# PGAN parameters
depths = [256, 256, 128, 64, 32, 16]
init_resolution_size=(8,5)
# init_resolution_size=(2,2)
num_epochs = 60
num_epochs_per_resolution = int(num_epochs / len(depths))
negative_slope = 0.2
fade_in_percentage = 0.5
save_interval = 50

normalization = True
mini_batch_normalization = False

# WGAN parameters
n_critic = 5
c = 0.01

# WGAN-GP parameters
# lambda
# beta1 = 0.5

# dataset
base_directory = "../data/raw/Bass"
# base_directory = "../data/raw/Single-bass"
# base_directory = "../data/raw/nsynth-train/audio"
# base_directory = "../data/raw/Bass-train"
