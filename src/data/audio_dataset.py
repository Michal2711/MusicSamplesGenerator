import sys
sys.path.append('../')
from features.Pipeline import Pipeline

import glob
import os
import torch
from torch.utils.data import Dataset

class AudioSpectrogramDataset(Dataset):
    def __init__(self, base_directory, spectro_type):
        self.base_directory = base_directory
        self.spectro_type = spectro_type
        self.file_list = glob.glob(os.path.join(base_directory, '*.wav'), recursive=True)
        self.pipeline = Pipeline(self.spectro_type)

        if spectro_type not in ['waveform', 'stft', 'mel', 'mfcc', 'cqt']:
            raise ValueError(
                "Spectrogram type incorrect. Possibilities: ['waveform', 'stft', 'mel',' mfcc', 'cqt']"
            )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_path = self.file_list[idx]

        spectrogram = self.pipeline.process(audio_path)

        if spectrogram.is_complex():
            spectrogram = spectrogram.to(dtype=torch.complex64).unsqueeze(0)
        else:
            spectrogram = torch.FloatTensor(spectrogram).unsqueeze(0)
        return spectrogram

    def get_data_size(self):
        return self.__getitem__(0).size()