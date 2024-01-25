import librosa
import numpy as np
import torch
import math
import json

base_sr = 16000

class Pipeline():
    def __init__(self, transform, resample_sr=base_sr, audio_length=base_sr, conditions_path=None) -> None:
        self.pre_pipeline = []
        self.post_pipeline = []
        self.resample_sr = resample_sr
        self.audio_length = audio_length
        self.conditions_path = conditions_path

        if self.conditions_path is not None:
            self.load_conditions()
            self.pitch_values = [i for i in range(9,110)]
            self.velocity_values = [25, 50, 75, 100, 127]

        self.transform = transform
        {
            "waveform": self.create_wave_pipeline,
            "stft": self.create_stft_pipeline,
            "mel": self.create_mel_pipeline,
            "mfcc": self.create_mfcc_pipeline,
            "cqt": self.create_cqt_pipeline,
        }[transform]()

    def create_wave_pipeline(self):
        self.pre_pipeline = [
            self.resample,
            self.padding,
            self.fade,
            self.normalization,
            self.transform_to_torch
        ]

        self.post_pipeline = [
            self.transform_to_numpy,
            self.denormalization,
        ]

    def create_stft_pipeline(self):
        self.pre_pipeline = [
            self.resample,
            self.padding,
            self.fade,
            self.normalization,
            self.stft,
            self.transform_to_torch,
            self.cut
        ]

        self.post_pipeline = [
            self.transform_to_numpy,
            self.istft,
            self.denormalization,
        ]

    def create_mel_pipeline(self):
        self.pre_pipeline = [
            self.mel,
            self.transform_to_torch,
            self.padding_columns,
        ]

        self.post_pipeline = [
            self.depadding_columns,
            self.transform_to_numpy,
            self.mel_invers,
        ]

    def create_mfcc_pipeline(self):
        self.pre_pipeline = [
            self.resample,
            self.padding,
            self.fade,
            self.normalization,
            self.mfcc,
            self.transform_to_torch
        ]

        self.post_pipeline = [
            self.transform_to_numpy,
            self.imfcc,
            self.denormalization
        ]

    def create_cqt_pipeline(self):
        self.pre_pipeline = [
            self.resample,
            self.padding,
            self.fade,
            self.normalization,
            self.cqt,
            self.transform_to_torch
        ]

        self.post_pipeline = [
            self.transform_to_numpy,
            self.icqt,
            self.denormalization
        ]

    def process(self, audio_path):
        audio = self.loader(audio_path)
        for func in self.pre_pipeline:
            audio = func(audio)
        if self.conditions_path is not None:
            conditional_vector = self.get_conditions(audio_path)
            return audio, torch.tensor(conditional_vector)
        else:
            return audio, None

    def post_process(self, audio):
        for func in self.post_pipeline:
            audio = func(audio)
        return audio

    def load_conditions(self):
        with open(self.conditions_path) as f:
            self.conditions = json.load(f)

    def get_conditions(self, audio_path):
        audio_name_with_extension = audio_path.split('\\')[-1]
        sample_name = audio_name_with_extension.split('.')[0]
        audio_condition = self.conditions[sample_name]

        return self.create_conditional_vector(audio_condition)

    def create_conditional_vector(self, sample_attributes):
        """
            instrument_family - one-hot encoding
            instrument_source - one-hot encoding
            pitch - one-hot encoding
            velocity - one-hot encoding
            quality_vecor - vector with qualities
        """

        instrument_family_vector = [1 if i == sample_attributes['instrument_family'] else 0 for i in range(11)]
        # instrument_source_vector = [1 if i == sample_attributes['instrument_source'] else 0 for i in range(3)]
        # pitch_vector = [1 if p == sample_attributes['pitch'] else 0 for p in self.pitch_values]
        # velocity_normalized = self.velocity_values.index(sample_attributes['velocity']) / (len(self.velocity_values) - 1)
        # velocity_vector = [1 if v == sample_attributes['velocity'] else 0 for v in self.velocity_values]
        # qualities_vector = sample_attributes['qualities']

        conditional_vector = instrument_family_vector

        # conditional_vector = pitch_vector + velocity_vector + qualities_vector

        return conditional_vector

    def loader(self, audio_path):
        audio, sr = librosa.core.load(
            path=audio_path,
            sr=self.resample_sr
        )

        self.input_sr = sr
        return audio

    def resample(self, audio):
        return librosa.core.resample(y=audio, target_sr=self.resample_sr, orig_sr=self.input_sr)

    def padding(self, audio):
        if len(audio) < self.audio_length:
            return np.append(audio, np.zeros(self.audio_length-len(audio)))
        else:
            return audio

    def padding_columns(self, tensor):
        if math.log2(tensor.shape[1] ).is_integer() is False:
            new_tensor = torch.zeros(tensor.shape[0], 128)
            new_tensor[:, :tensor.shape[1]] = tensor
            return new_tensor

    def depadding_columns(self, tensor):
        return tensor[:, :126]

    def normalization(self, audio):
        self.min_val, self.max_val = audio.min(), audio.max()
        normalized_mel = (audio - self.min_val) / (self.max_val - self.min_val)
        normalized_mel = normalized_mel * 2 - 1
        return normalized_mel

    
    def denormalization(self, audio):
        denormalized_mel = (audio + 1) / 2
        denormalized_mel = denormalized_mel * (self.max_val - self.min_val) + self.min_val
        return denormalized_mel

    def fade(self, audio, percent=30.):
        fade_idx = int(audio.shape[-1] * percent / 100.)
        fade_curve = np.logspace(1, 0, fade_idx)
        fade_curve -= min(fade_curve)
        fade_curve /= max(fade_curve)
        audio[-fade_idx:] *= fade_curve
        return audio

    def cut(self, spectrogram):
        spectrogram = spectrogram[:, :160]
        return spectrogram

    def mel(self, wave):
        return librosa.feature.melspectrogram(
            y=wave,
            sr = base_sr,
            n_fft=getattr(self, 'n_fft', 2048),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 2048),
            n_mels=getattr(self, 'n_mels', 256),
        )
    
    def mel_invers(self, mel_spec):
        return librosa.feature.inverse.mel_to_audio(
            M=mel_spec,
            sr=base_sr,
            n_fft=getattr(self, 'n_fft', 2048),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 2048),
        )
    
    def stft(self, wave):
        return librosa.core.stft(
            y=wave,
            n_fft=getattr(self, 'n_fft', 2048),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 2048)
        )
    
    def istft(self, stft_spec):
        return librosa.core.istft(
            stft_matrix=stft_spec,
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 2048),
        )
    
    def mfcc(self, wave):
        return librosa.feature.mfcc(
            y = wave,
            sr = self.resample_sr,
            n_fft = getattr(self, 'n_fft', 2048),
            n_mels = getattr(self, 'n_mel', 128),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 1024),
            S=None,
            n_mfcc=getattr(self, 'n_mfcc', 20),
            dct_type=2,
            norm='ortho',
            lifter=0
        )
    
    def imfcc(self, mfcc):
        return librosa.feature.inverse.mfcc_to_audio(
            mfcc = mfcc,
            n_fft=getattr(self, 'fft_size', 2048),
            n_mels=getattr(self, 'n_mel', 128),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_size', 1024),
            dct_type=2,
            norm='ortho',
            ref=1.0
        )

    def cqt(self, wave):
        return librosa.core.cqt(
            y=wave,
            hop_length=getattr(self, 'hop_length', 512),
            n_bins=getattr(self, 'n_bins', 84),
            bins_per_octave=getattr(self, 'bins_per_octave', 12)
        )

    def icqt(self, cqt):
        return librosa.core.icqt(
            C=cqt,
            hop_length=getattr(self, 'hop_length', 512)
        )

    def transform_to_torch(self, audio):
        if type(audio) is np.ndarray:
            if np.iscomplexobj(audio):
                return torch.tensor(audio, dtype=torch.complex64)
            return torch.from_numpy(audio)
        elif type(audio) is torch.Tensor:
            return audio.float()
        else:
            return torch.FloatTensor(audio)
    
    def transform_to_numpy(self, audio):
        return audio.numpy()