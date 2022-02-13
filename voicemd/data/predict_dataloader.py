import torch
from torch.utils.data import DataLoader

from voicemd.data.process_sound import compute_specgram, load_waveform


def make_predict_dataloader(sound_filename, hyper_params):
    """Useful for prediction on single file"""

    predict_dataset = PredictDataset(
        sound_filename=sound_filename,
        spec_type=hyper_params["spec_type"],
        window_len=hyper_params["window_len"],
        in_channels=hyper_params["in_channels"],
        preprocess=True,
        normalize=hyper_params["normalize_spectrums"],
    )

    predict_dataloader = DataLoader(
        predict_dataset, batch_size=hyper_params["batch_size"], shuffle=False
    )

    return predict_dataloader


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
