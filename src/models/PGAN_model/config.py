# # hyperparameters

# type parameters
get_type = 'audio'
init_resolution_size=(8,5)
gen_output_dim=1

# get_type = 'pictures'
# init_resolution_size=(2,2)
# gen_output_dim=3

# Basic model parameters
latent_dim = 256
output_dim=1
learning_rate = 5e-4
loss = 'MSE'
batch_size = 64
gpu = True

# PGAN parameters
depths = [256, 256, 128, 64, 32, 16]
num_epochs = 60
num_epochs_per_resolution = int(num_epochs / len(depths))
negative_slope = 0.2
fade_in_percentage = 0.5
save_interval = 50

normalization = False
mini_batch_normalization = False

# dataset
# base_directory = "../data/raw/Bass"
base_directory = "../data/raw/Single-bass"
# base_directory = "../data/raw/nsynth-train/audio"
# base_directory = "../data/raw/Bass-train"
