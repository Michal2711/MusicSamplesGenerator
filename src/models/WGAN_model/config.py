# # hyperparameters

# type parameters
get_type = 'audio'
init_resolution_size=(32,5)
gen_output_dim=1

# get_type = 'pictures'
# init_resolution_size=(2,2)
# gen_output_dim=3

# Basic model parameters
latent_dim = 256
output_dim=1
learning_rate = 5e-5 # 5e-5 9.5e-5
loss = 'WGAN'
batch_size = 64
gpu = True
epsilon_D = 0.001

# PGAN parameters
depths = [256, 256, 128, 64, 32, 16]
num_epochs = 300
num_epochs_per_resolution = int(num_epochs / len(depths))
negative_slope = 0.2
fade_in_percentage = 0.5
save_interval = 10

normalization = False
mini_batch_normalization = False

# WGAN parameters
n_critic = 3 # 5
c = 0.01

# WGAN-GP parameters
# lambda
# beta1 = 0.5

# dataset
# base_directory = "../data/raw/Bass"
base_directory = "../data/raw/Single-bass"
# base_directory = "../data/raw/nsynth-train/audio"
# base_directory = "../data/raw/Bass-train"
