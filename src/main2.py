import streamlit as st
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

# pretrained_path = './../models/saved_models/WPGAN/WPGAN-GP-single-best'
pretrained_path = './../models/saved_models/WPGAN/WPGAN-GP-acgan-best1'
# pretrained_path = './../models/saved_models/WPGAN/WPGAN-GP-acgan-test-set'

def load_model(model_path):
    pretrained_Music_WPGAN = WPGAN_GP(
    latent_dim=latent_dim,
    output_dim=output_dim,
    lr=learning_rate,
    loss=loss,
    batch_size=batch_size,
    gpu=gpu,
    depths=depths, 
    negative_slope=negative_slope,
    fade_in_percentage=fade_in_percentage,
    save_interval=save_interval,
    normalization=normalization,
    mini_batch_normalization=mini_batch_normalization,
    epsilon_D=epsilon_D,
    gen_type=get_type,
    init_resolution_size=init_resolution_size,
    num_epochs_per_resolution=num_epochs_per_resolution,
    gen_output_dim=gen_output_dim,
    c=c,
    n_critic=n_critic
    )
    pretrained_Music_WPGAN.load_pretrained_model(model_path, load_optimizers=True)
    return pretrained_Music_WPGAN

def generate_spectrogram_from_pretrained(pretrained_model, feature_vector):
    pretrained_model.generator.eval()

    with torch.no_grad():
        z = torch.randn(1, pretrained_model.latent_dim, 1, 1).to(pretrained_model.device)
        if feature_vector is not None:
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

def create_feature_vector(selected_instrument, instrument_options):

    instrument_vector = [1 if instrument == selected_instrument else 0 for instrument in instrument_options]

    feature_vector = instrument_vector
    feature_vector = torch.tensor(feature_vector)

    return feature_vector

@st.cache_resource
def cached_load_model(model_path):
    return load_model(model_path)

def main():
    st.title('Mel Spectrogram Generator')

    model = cached_load_model(pretrained_path)

    instrument_options = ['bass', 'brass', 'flute', 'guitar', 'keyboard', 'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal']
    selected_instrument = st.selectbox('Select Velocity', instrument_options)

    if 'feature_vector' not in st.session_state:
        st.session_state['feature_vector'] = None

    if 'original_spectrogram' not in st.session_state:
        st.session_state['original_spectrogram'] = None

    if 'modified_spectrogram' not in st.session_state:
        st.session_state['modified_spectrogram'] = None

    if st.button('Save conditions'):
        st.session_state['feature_vector'] = create_feature_vector(selected_instrument, instrument_options)

    if st.button('Generate Spectrogram'):
        st.session_state['original_spectrogram'] = generate_spectrogram_from_pretrained(model, st.session_state['feature_vector'])
        st.session_state['modified_spectrogram'] = None

    spectrogram_to_display = st.session_state['modified_spectrogram'] if st.session_state['modified_spectrogram'] is not None else st.session_state['original_spectrogram']
    if spectrogram_to_display is not None:
        # fig = plot_spectrogram(spectrogram_to_display)
        # st.pyplot(fig)
        fig = plot_spectrogram_with_plotly(spectrogram_to_display)
        st.plotly_chart(fig, use_container_width=True)

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