import streamlit as st
import json
import torch
import matplotlib.pyplot as plt
import plotly.express as px
import librosa
import numpy as np
import io
import soundfile as sf

import sys
sys.path.append('../src')
from models.WGAN_model.WPGAN_GP import WPGAN_GP
from models.WGAN_model.config import *

@st.cache_resource
def load_model_configs():
    with open('./../models/saved_models/pre_trained_config.json', 'r') as f:
        model_configs = json.load(f)
    return model_configs

def load_model(model_config):

    pretrained_Music_WPGAN = WPGAN_GP(
        latent_dim=model_config['basic_model_parameters']['latent_dim'],
        output_dim=model_config['basic_model_parameters']['output_dim'],
        lr=model_config['basic_model_parameters']['learning_rate'],
        loss=model_config['basic_model_parameters']['loss'],
        batch_size=model_config['basic_model_parameters']['batch_size'],
        gpu=model_config['basic_model_parameters']['gpu'],
        depths=model_config['PGAN_parameters']['depths'],
        negative_slope=model_config['basic_model_parameters']['negative_slope'],
        fade_in_percentage=model_config['PGAN_parameters']['fade_in_percentage'],
        save_interval=model_config['basic_model_parameters']['save_interval'],
        normalization=model_config['basic_model_parameters']['normalization'],
        mini_batch_normalization=model_config['basic_model_parameters']['mini_batch_normalization'],
        epsilon_D=model_config['basic_model_parameters']['epsilon_D'],
        gen_type=model_config['type_parameters']['get_type'],
        init_resolution_size=model_config['type_parameters']['init_resolution_size'],
        num_epochs_per_resolution=model_config['PGAN_parameters']['num_epochs_per_resolution'],
        gen_output_dim=model_config['basic_model_parameters']['gen_output_dim'],
        c=model_config['WGAN_parameters']['c'],
        n_critic=model_config['WGAN_parameters']['n_critic'],
        lambda_gp=model_config['WGAN-GP_parameters']['lambda_gp'],
        acgan=model_config['ACGAN_parameters']['acgan'],
        feature_size=model_config['ACGAN_parameters']['feature_size'],
        features_keys_order=model_config['ACGAN_parameters']['features_keys_order']
    )

    pretrained_path = model_config['pretrained_path']
    pretrained_Music_WPGAN.load_pretrained_model(pretrained_path, load_optimizers=True)
    return pretrained_Music_WPGAN

def create_feature_vector(selected_features, features_keys_order):
    feature_vector = []
    for feature_name, feature_info in sorted(features_keys_order.items(), key=lambda item: item[1]['order']):
        if feature_name in selected_features:
            selected_option = selected_features[feature_name]
            if isinstance(selected_option, list):
                feature_vector += [1 if option in selected_option else 0 for option in feature_info['values']]
            else:
                feature_vector += [1 if option == selected_option else 0 for option in feature_info['values']]
        else:
            raise ValueError(f"Feature {feature_name} is not in the selected features.")
    return torch.tensor(feature_vector)

def generate_spectrogram_from_pretrained(pretrained_model, feature_vector):
    pretrained_model.generator.eval()

    with torch.no_grad():
        z = torch.randn(1, pretrained_model.latent_dim, 1, 1).to(pretrained_model.device)
        if pretrained_model.acgan and feature_vector.nelement() != 0:
            feature_vector = feature_vector.view(1, -1, 1, 1).to(pretrained_model.device)
            z = torch.cat((z, feature_vector), dim=1)
        generated_spectrogram = pretrained_model.generator(z)
        spectrogram = generated_spectrogram.cpu().detach()
        spectrogram = spectrogram.squeeze()

        return spectrogram
    
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots()

    S_dB = librosa.amplitude_to_db(spectrogram, ref=np.min)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=16000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='WGAN generated mel-spectrogram')

    return fig

def plot_spectrogram_with_plotly(spectrogram, sr=22050, hop_length=512):
    S_dB = librosa.amplitude_to_db(spectrogram, ref=np.min)

    time_axis = np.linspace(0, spectrogram.shape[1] * hop_length / sr, spectrogram.shape[1])
    freq_axis = np.linspace(0, sr / 2, spectrogram.shape[0])

    fig = px.imshow(S_dB, x=time_axis, y=freq_axis, color_continuous_scale='electric', 
                    labels=dict(color="Magnitude"), 
                    aspect='auto', origin='lower')

    fig.update_layout(
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)'
    )

    fig.update_layout(coloraxis_colorbar=dict(title='dB'))

    return fig

def get_audio_from_melspectrogram(melspectrogram, sr=16000, n_fft=2048, hop_length=512, win_length=2048):
    audio = librosa.feature.inverse.mel_to_audio(
        M=melspectrogram.numpy(),
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    return audio

def freq_to_mel(freq, sr=16000, n_mels=256):
    mel_frequencies = librosa.mel_frequencies(n_mels=n_mels, fmax=sr / 2)
    index = np.argmin(np.abs(mel_frequencies - freq))
    return index

def modify_spectrogram(spectrogram, freq_range, sr=16000, n_mels=256):
    start_idx = freq_to_mel(freq_range[0], sr, n_mels)
    end_idx = freq_to_mel(freq_range[1], sr, n_mels)

    start_idx = max(0, min(start_idx, spectrogram.shape[0] - 1))
    end_idx = max(0, min(end_idx, spectrogram.shape[0] - 1))

    if start_idx < end_idx:
        spectrogram[start_idx:end_idx, :] = 0
    return spectrogram

@st.cache_resource
def cached_load_model(model_path):
    return load_model(model_path)

def main():
    st.title('Mel Spectrogram Generator')

    model_configs = load_model_configs()
    last_model_key = st.session_state.get('last_model_key', None)

    selected_model_key = st.selectbox('Select Model Version', list(model_configs.keys()))

    if selected_model_key != last_model_key:
        st.session_state['original_spectrogram'] = None
        st.session_state['modified_spectrogram'] = None
        st.session_state['feature_vector'] = None

        st.session_state['last_model_key'] = selected_model_key

    selected_model_config = model_configs[selected_model_key]
    model = cached_load_model(selected_model_config)

    if model.acgan:
        selected_features = {}
        for feature_name, feature_info in model.features_keys_order.items():
            if feature_name == 'pitch':
                selected_pitch = st.number_input('Select Pitch High', min_value=feature_info['values'][0], max_value=feature_info['values'][-1], value=feature_info['values'][0])
                selected_features[feature_name] = selected_pitch
            elif feature_name == 'velocity':
                selected_velocity = st.selectbox('Select Velocity', feature_info['values'])
                selected_features[feature_name] = selected_velocity
            elif feature_name == 'qualities':
                selected_qualities = st.multiselect('Select Qualities (max 5)', feature_info['values'])
                selected_features[feature_name] = selected_qualities
            elif feature_name == 'instrument':
                selected_instrument = st.selectbox('Select Instrument', feature_info['values'])
                selected_features[feature_name] = selected_instrument

    if model.acgan and 'pitch' in selected_features and 'bright' in selected_qualities and 'dark' in selected_qualities:
        st.error("You cannot choose both 'bright' and 'dark'. Please select one.")

    if model.acgan and 'qualities' in selected_features and 'fast_decay' in selected_qualities and 'long_release' in selected_qualities:
        st.error("You cannot choose both 'fast decay' and 'long release'. Please select one.")

    if model.acgan and 'qualities' in selected_features and len(selected_qualities) > 5:
        st.error('Please select no more than 5 qualities.')
        selected_qualities = selected_qualities[:5]

    if 'feature_vector' not in st.session_state:
        st.session_state['feature_vector'] = None

    if 'original_spectrogram' not in st.session_state:
        st.session_state['original_spectrogram'] = None

    if 'modified_spectrogram' not in st.session_state:
        st.session_state['modified_spectrogram'] = None

    if model.acgan and st.button('Save conditions'):
        st.session_state['feature_vector'] = create_feature_vector(selected_features, model.features_keys_order)

    if st.button('Generate Spectrogram'):
        st.session_state['original_spectrogram'] = generate_spectrogram_from_pretrained(model, st.session_state['feature_vector'])
        st.session_state['modified_spectrogram'] = None

    spectrogram_to_display = st.session_state['modified_spectrogram'] if st.session_state['modified_spectrogram'] is not None else st.session_state['original_spectrogram']
    if spectrogram_to_display is not None:
        fig = plot_spectrogram(spectrogram_to_display)
        st.pyplot(fig)
        # fig = plot_spectrogram_with_plotly(spectrogram_to_display)
        # st.plotly_chart(fig, use_container_width=True)

        audio = get_audio_from_melspectrogram(spectrogram_to_display)
        buffer = io.BytesIO()
        sf.write(buffer, audio, 16000, format='WAV')
        buffer.seek(0)
        st.audio(buffer, format='audio/wav')

    if st.session_state['original_spectrogram'] is not None:
        mel_bins = librosa.mel_frequencies(n_mels=st.session_state['original_spectrogram'].shape[0], fmax=16000/2)
        min_freq, max_freq = mel_bins[0], mel_bins[-1]

        freq_range = st.slider('Select Frequency Range to Erase', min_value=int(min_freq), max_value=int(max_freq), value=(100, 200))

        st.button('Modify mel spectrogram', on_click=modify_spectrogram_callback, args=(freq_range,))
        st.button('Revert to original', on_click=revert_spectrogram_callback)

def modify_spectrogram_callback(freq_range):
    sr = 16000
    n_mels = st.session_state['original_spectrogram'].shape[0]
    st.session_state['modified_spectrogram'] = modify_spectrogram(st.session_state['original_spectrogram'].clone(), freq_range, sr, n_mels)

def revert_spectrogram_callback():
    st.session_state['modified_spectrogram'] = None

if __name__ == "__main__":
    main()