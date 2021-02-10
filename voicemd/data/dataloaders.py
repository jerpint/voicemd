import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch

from voicemd.data.process_sound import compute_specgram, load_waveform

logger = logging.getLogger(__name__)

class AudioDataset(torch.utils.data.Dataset):
    """Voice recordings dataset."""

    def __init__(self,
                 metadata,
                 voice_clips_dir,
                 spec_type,
                 in_channels=1,
                 window_len=128,
                 normalize=False,
                 preprocess=False,
                 dev=False,
                 dev_step_size=64,
                 transform=None,
                 split=None,
                ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            voice_clips_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.metadata = metadata
        self.voice_clips_dir = voice_clips_dir
        self.spec_type = spec_type
        self.transform = transform
        self.in_channels = in_channels
        self.window_len = window_len
        self.dev = False # if in dev, the entire spectrum is processed
        self.dev_step_size = dev_step_size
        self.normalize = normalize
        self.preprocess = preprocess
        self.split = split

        self._compute_specgram = compute_specgram
        self._load_waveform = load_waveform

        # If preprocess it will preprocess and cache each spec
        # This will speed up computation but wont scale if the dataset
        # gets larger
        if self.preprocess:
            self._preprocess_dataset()

    def _specgram_from_uid(self, uid):
        '''Retrieve a specgram from the patient's UID'''

        fname = self.metadata.loc[uid]['filename']
        full_path = Path(self.voice_clips_dir, fname)
        waveform, sr = self._load_waveform(full_path)
        specgram = self._compute_specgram(waveform, sr, self.spec_type, self.normalize)

        return specgram


    def _preprocess_dataset(self):
        '''Cache the dataset in RAM'''

        # disable tqdm output for validation and test sets, False when train
        disable_tqdm = not(self.split == 'train')
        self.specs = {
            uid: self._specgram_from_uid(uid) for uid in tqdm(
                self.metadata.index,
                disable=disable_tqdm
            )
        }

        self.labels = {uid: self.metadata.loc[uid] for uid in self.metadata.index}

    def show_sample(self, sample):
        '''For visualization only'''

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 4))
        cax = ax.matshow(torch.squeeze(sample[0]), interpolation='nearest', aspect='auto', cmap=plt.cm.afmhot, origin='lower')
        fig.colorbar(cax)
        plt.title('Spectrogram')
        plt.show()
        return fig, ax


class TrainDataset(AudioDataset):

    def __len__(self):

        return len(self.metadata)

    def __getitem__(self, idx):

        # Pick a random valid index to sample at
        # Sample a spectrum at random from the entire spectrum
        uid = self.metadata.index[idx]
        start_idx = np.random.randint(0, self.specs[uid].shape[2]-self.window_len)
        spec = self.specs[uid][..., start_idx:start_idx+self.window_len]

        # Expand on dims if necessary (for a model architecture expecting e.g. rbg)
        if self.in_channels != 1:
            spec = spec.expand(self.in_channels, spec.shape[1], spec.shape[2])

        label = float(self.labels[uid]['gender'] == 'M')

        return spec, label


class EvalDataset(AudioDataset):

    def __len__(self):

        uid = self.metadata.index[0]

        #TODO: Update this
        return (self.specs[uid].shape[2] - self.window_len) // self.dev_step_size

    def __getitem__(self, idx):

        uid = self.metadata.index[0]

        # Sample a spectrogram at every dev_step_size, equivalent to hop length
        idx_spec = idx*self.dev_step_size
        spec = self.specs[uid][..., idx_spec:idx_spec+self.window_len]

        # Expand on dims if necessary (for a model architecture expecting e.g. rbg)
        if self.in_channels != 1:
            spec = spec.expand(self.in_channels, spec.shape[1], spec.shape[2])

        gender = float(self.labels[uid]['gender'] == 'M')

        return spec, gender


class PredictDataset(torch.utils.data.Dataset):
    '''To be used for predicting single sounds'''

    def __init__(self,
                 sound_filename,
                 spec_type,
                 in_channels,
                 window_len,
                 normalize,
                 preprocess,
                 transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            voice_clips_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.sound_filename = sound_filename
        self.spec_type = spec_type
        self.transform = transform
        self.in_channels = in_channels
        self.window_len = window_len
        self.normalize = normalize
        self.preprocess = preprocess

        self._compute_specgram = compute_specgram
        self._load_waveform = load_waveform

        self.waveform, self.sr = self._load_waveform(self.sound_filename)
        self.specgram = self._compute_specgram(self.waveform, self.sr, self.spec_type, self.normalize)

    def __len__(self):

        '''computes the result on all possible spectrograms'''
        return self.specgram.shape[2] - self.window_len

    def __getitem__(self, idx):

        spec = self.specgram[..., idx:idx+self.window_len]
        if self.in_channels != 1:
            spec = spec.expand(self.in_channels, spec.shape[1], spec.shape[2])

        return spec
