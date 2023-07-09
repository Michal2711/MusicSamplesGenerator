import sys
sys.path.append('../')
from features.build_features import MelPipeline

import glob
import os
import torch
from torch.utils.data import Dataset
import librosa


class AudioSpectrogramDataset(Dataset):
    def __init__(self, base_directory, spectro_type):
        self.base_directory = base_directory
        self.spectro_type = spectro_type
        self.file_list = glob.glob(os.path.join(base_directory, '*.wav'), recursive=True)

        if spectro_type not in ['spectrogram', 'mel-spectrogram']:
            raise ValueError(
                "Spectrogram type incorrect. Possibilities: ['spectrogram', 'mel-spectrogram']"
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.file_list[idx]

        audio, sr= librosa.load(audio_path,
                                mono=True)

        # spectrogram
        if self.spectro_type == "spectrogram":
            spectrogram = librosa.stft(audio)
            spectrogram = torch.from_numpy(spectrogram)
        else:

            self.pipeline = MelPipeline()

            spectrogram = self.pipeline(audio_path)

        spectrogram = spectrogram.unsqueeze(0)

        return spectrogram
