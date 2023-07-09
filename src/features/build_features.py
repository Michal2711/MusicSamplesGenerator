import torch
import numpy as np
import librosa

class MelPipeline(torch.nn.Module):
    def __init__(
        self,
        resample_sr = 22050,
        n_fft=1024,
        n_mels=256,
        audio_length = 22050,
        hop_length=256,
        win_length=1024
    ) -> None:
        super().__init__()
        self.resample_sr = resample_sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.audio_length = audio_length
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, audio_path) -> torch.Tensor:

        waveform = self.loader(audio_path)

        resampled_wave = librosa.core.resample(y=waveform, target_sr=self.resample_sr, orig_sr=self.input_sr)

        padded_wave = self.padding(resampled_wave)

        faded_wave = self.fade(padded_wave)

        normalized_wave = self.normalization(faded_wave)

        mel_spectro = self.mel(normalized_wave)

        mel_spectro = torch.from_numpy(mel_spectro)

        return mel_spectro

    def loader(self, audio_path):
        audio, sr = librosa.core.load(
            path=audio_path,
            mono=True
        )

        self.input_sr = sr

        return audio

    def padding(self, audio):
        if len(audio) < self.audio_length:
            return np.append(audio, np.zeros(self.audio_length-len(audio)))
        else:
            return audio

    def normalization(self, audio):
        if max(audio) != 0:
            return audio/max(audio)
        else:
            return audio

    def fade(self, audio, percent=30.):
        fade_idx = int(audio.shape[-1] * percent / 100.)
        fade_curve = np.logspace(1, 0, fade_idx)
        fade_curve -= min(fade_curve)
        fade_curve /= max(fade_curve)
        audio[-fade_idx:] *= fade_curve
        return audio

    def mel(self, wave):
        return librosa.feature.melspectrogram(
            y=wave,
            sr = self.resample_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels = self.n_mels
        )
