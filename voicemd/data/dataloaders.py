from pathlib import Path

import pandas as pd
import torch
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    """Voice recordings dataset."""

    def __init__(self, metadata, voice_clips_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            voice_clips_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = metadata
        self.voice_clips_dir = voice_clips_dir
        self.transform = transform
        self.spectrograms = []

        self._preprocess()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #  fname = self.metadata['filename'][idx]
        #  spec, w, sr = self._load_spectrogram(fname)
        spec = self.specs[idx]
        label = self.labels[idx]

        return spec, label

    def _load_spectrogram(self, fname):
        full_path = Path(self.voice_clips_dir, fname)
        waveform, sr = torchaudio.load(full_path)  # load tensor from file
        #  specgram = torchaudio.transforms.Spectrogram()(waveform)
        specgram = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=40)(waveform)

        return specgram, waveform, sr

    def _preprocess(self):

        self.specs = [
            self._load_spectrogram(ii[1])[0]
            for ii in self.metadata["filename"].iteritems()
        ]
        self.labels = [row[1] for row in self.metadata.iterrows()]

        pass


if __name__ == "__main__":

    from voicemd.utils.utils import load_yaml_config

    root_dir = "/home/jerpint/voicemd/"
    dpath = root_dir + "data/"
    fname = dpath + "voice_clips/cleaned_metadata.csv"
    voice_clips_dir = dpath + "voice_clips/"
    config = load_yaml_config("config.yaml")
    metadata = pd.read_csv(fname)
    metadata = metadata[metadata["filename"].notna()]
    audio_dataloader = AudioDataset(metadata, voice_clips_dir)
