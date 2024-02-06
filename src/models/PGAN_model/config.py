# type parameters
get_type = 'audio'
init_resolution_size=(8,4)
gen_output_dim=1

# Basic model parameters
latent_dim = 256
output_dim=1
learning_rate = 5e-5
loss = 'MSE'
batch_size = 16
gpu = True
epsilon_D = 0.001

# PGAN parameters
depths = [256, 256, 128, 64, 32, 16]
num_epochs = 1200
i = int(num_epochs / len(depths))
num_epochs_per_resolution = [i, 2*i, 2*i, 3*i, 4*i, 5*i]
negative_slope = 0.2
fade_in_percentage = 0.5
save_interval = 5

normalization = False
mini_batch_normalization = True


# WITHOUT ACGAN
# acgan=False
# feature_size = 0
# features_keys_order = None

# WITH ACGAN
acgan=True
feature_size = 11
features_keys_order = {
    'instrument': 
    {
        'order': 0, 
        'values': ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    }
}

# feature_size = 116
# features_keys_order = {
#     'pitch': 
#     {
#         'order': 0, 
#         'values': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
#     }, 
#     'velocity': 
#     {
#         'order': 1, 
#         'values': ['25', '50', '75', '100', '127']
#     }, 
#     'qualities': 
#     {
#         'order': 2, 
#         'values': ['dark', 'distortion', 'reverb', 'fast_decay', 'percussive', 'tempo-synced', 'bright', 'multiphonic', 'long_release', 'nonlinear_env']
#     }
# }

# DATASET
# base_directory = "../data/raw/Bass"
base_directory = "../data/raw/Single-bass/audio"
# base_directory = "../data/raw/Single-piano/audio"
# base_directory = "../data/raw/nsynth-test/audio"
# base_directory = "../data/raw/Bass-train"
# base_directory = "../data/raw/Bass_Keyboard/audio"

# conditions
conditions_path = '../data/raw/Single-bass/examples.json'
# conditions_path = '../data/raw/Single-piano/examples.json'
# conditions_path = '../data/raw/nsynth-train/examples.json'
# conditions_path = '../data/raw/nsynth-test/examples.json'
# conditions_path = "../data/raw/Bass_Keyboard/examples.json"
# conditions_path = None