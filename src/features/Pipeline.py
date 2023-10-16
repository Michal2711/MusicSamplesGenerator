import librosa
import numpy as np
import torch

class Pipeline():
    def __init__(self, transform, resample_sr=22050, audio_length=22050) -> None:
        self.pre_pipeline = []
        self.post_pipeline = []
        self.resample_sr = resample_sr
        self.audio_path = ""
        self.audio_length = audio_length

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
            self.transform_to_torch
        ]

        self.post_pipeline = [
            self.transform_to_numpy,
            self.istft,
            self.denormalization,
        ]

    def create_mel_pipeline(self):
        self.pre_pipeline = [
            self.resample,
            self.padding,
            self.fade,
            self.normalization,
            self.mel,
            self.transform_to_torch,
            self.cut
        ]

        self.post_pipeline = [
            self.transform_to_numpy,
            self.mel_invers,
            self.denormalization,
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
        self.audio_path = audio_path
        audio = self.loader(self.audio_path)
        for func in self.pre_pipeline:
            audio = func(audio)
        return audio

    def post_process(self, audio):
        for func in self.post_pipeline:
            audio = func(audio)
        return audio

    def loader(self, audio_path):
        audio, sr = librosa.core.load(
            path=audio_path,
            mono=True,
            # offset=0.0,
            # # duration=self.audio_length/self.input_sr,
            # dtype=np.float32,
            # res_type='kaiser_best'
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

    def normalization(self, audio):
        return librosa.util.normalize(audio)
    
    def denormalization(self, audio):
        return audio * np.max(np.abs(audio))

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
            # sr = self.resample_sr,
            n_fft=getattr(self, 'n_fft', 1024),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 1024),
            n_mels =getattr(self, 'n_mels', 256)
        )
    
    def mel_invers(self, mel_spec):
        return librosa.feature.inverse.mel_to_audio(
            M=mel_spec,
            # sr=self.resample_sr,
            n_iter=100,
            n_fft=getattr(self, 'n_fft', 1024),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 1024),
        )
    
    def stft(self, wave):
        return librosa.core.stft(
            y=wave,
            n_fft=getattr(self, 'n_fft', 1024),
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 1024)
        )
    
    def istft(self, stft_spec):
        return librosa.core.istft(
            stft_matrix=stft_spec,
            hop_length=getattr(self, 'hop_length', 512),
            win_length=getattr(self, 'win_length', 1024),
            # length=getattr(self, 'audio_length', 22050)
        )
    
    def mfcc(self, wave):
        return librosa.feature.mfcc(
            y = wave,
            # sr = self.resample_sr,
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
        # return torch.from_numpy(audio)
    
    def transform_to_numpy(self, audio):
        return audio.numpy()