Music Sample Generator
==============================

Introduction
------------
The Music Sample Generator is a project aimed at synthesizing mel spectrograms from short-duration audio samples (4 second duration). The core of this project is to compare different type of GAN models such as
* PGAN
* WGAN
* WGAN-GP

One of the standout features of this project is its support for conditional generation. Users can condition the model on specific attributes, allowing for the controlled generation of mel spectrograms that adhere to particular characteristics or criteria

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── audio_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── Pipeline.py
    │   │
    │   └── models         <- Scripts to train models and then use trained models to make
    │       │                 predictions
    │       ├── ACGAN      <- Implementation of ACGAN model
    │       ├── Base_model <- Implementation of BaseGAN model
    │       ├── PGAN_model <- Implementation of PGAN model based on BaseGAN model with Generator and Discriminator code and train code
    │       └── WGAN_model <- Implementation of WGAN and WGAN-GP models based on PGAN model
    │     
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

Installation 
------------

Install requirements

```
pip install -r requirements.txt
```

Running the application
------------
The model's interface is built using Streamlit, which allows for an interactive web application. To start the application, navigate to the src directory and run the Streamlit application using the following command:

```
cd src
streamlit run main.py
```

The dataset
------------

I trained my model based on Nsynth dataset which is available on link:
https://magenta.tensorflow.org/datasets/nsynth


Model Training
------------
I provided Jupyternotebooks for training different models:

- `6-test-PGAN.ipynb`: Notebook for training the PGAN model.
- `8-test-WGAN.ipynb`: Notebook for training the WGAN model.

These notebooks include detailed steps for creating model and training, from data preprocessing to generate samples. Configuration for training each model can be adjusted in respective config files:
* src/models/PGAN_model/config.py
* src/models/WGAN_model/config.py

Ensure to review and adjust the configuration parameters according to your system's capabilities and your project's requirements.

Example of config file:
------------

```
# type parameters
get_type = 'audio'
init_resolution_size=(8,4)
gen_output_dim=1

# Basic model parameters
latent_dim = 256
output_dim=1
learning_rate = 1e-4
loss = 'WGAN'
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

# WGAN parameters
n_critic = 3 # 5
c = 0.01

# WGAN-GP parameters
lambda_gp = 10


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
```

Pretrained models and generated samples
------------

Pretrained models and generated samples are located here:
* models/saved_models/
* models/generated_samples/

Also in models/saved_models/ dictionary there is a pre_trained_config.json file with configs for some pretrained models to use in Streamlit application. Pretrained path for Pitch-Velocity-Quality-condition should be replaced, because this models does not correspond to config parameters. This is merely a preparation of the possibility of using a model based on conditional features such as pitch, velocity and qualitative features. 