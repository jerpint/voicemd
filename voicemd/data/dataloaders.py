from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio


class AudioDataset(torch.utils.data.Dataset):
    """Voice recordings dataset."""

    def __init__(self, metadata, voice_clips_dir, in_channels=1, window_len=128, transform=None):
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
        self.in_channels = in_channels
        self.window_len = window_len
        self.spectrograms = []

        self._preprocess()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # This would work if we want to do on the fly preprocessing. Currently we are doing it in preprocessing and caching it in the ram as the dataset isnt too big
        #  fname = self.metadata['filename'][idx]
        #  spec, w, sr = self._load_spectrogram(fname)



        start_idx = np.random.randint(self.window_len, self.specs[idx].shape[2]-self.window_len)

        spec = self.specs[idx][..., start_idx:start_idx+self.window_len]
        if spec.shape[0] != 1:
            from IPython import embed; embed();
        if self.in_channels != 1:
            spec = spec.expand(self.in_channels, spec.shape[1], spec.shape[2])

        label = float(self.labels[idx]['gender'] == 'M')

        return spec, label

    def _load_spectrogram(self, fname):
        full_path = Path(self.voice_clips_dir, fname)
        waveform, sr = torchaudio.load(full_path)  # load tensor from file
        #  specgram = torchaudio.transforms.Spectrogram()(waveform)
        if waveform.shape[0] == 2:  # Assume Mono
            waveform = torch.unsqueeze(waveform[0, ...], dim=0)
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
    import torchvision.models as models

    root_dir = "/home/jerpint/voicemd/"
    dpath = root_dir + "data/"
    fname = dpath + "voice_clips/cleaned_metadata.csv"
    voice_clips_dir = dpath + "voice_clips/"
    config = load_yaml_config("config.yaml")
    metadata = pd.read_csv(fname)
    metadata = metadata[metadata["filename"].notna()]
    audio_dataloader = AudioDataset(metadata, voice_clips_dir)

    densenet = models.densenet121(pretrained=False)
    #  densenet._modules.pop('classifier')
    densenet = torch.nn.Sequential(densenet, torch.nn.Linear(in_features=1000, out_features=2))
