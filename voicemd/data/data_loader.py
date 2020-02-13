import os
import pandas as pd

import numpy
from torch.utils.data import Dataset, DataLoader
from voicemd.data.dataloaders import AudioDataset

# __TODO__ change the dataloader to suit your needs...


def load_data(args, hyper_params):
    # __TODO__ load the data
    fname = args.data + "cleaned_metadata.csv"
    metadata = pd.read_csv(fname)
    train_data = AudioDataset(
        metadata[0:200], voice_clips_dir=args.data, transform=None
    )
    dev_data = AudioDataset(metadata[200:], voice_clips_dir=args.data, transform=None)

    train_loader = DataLoader(
        train_data, batch_size=hyper_params["batch_size"], shuffle=True
    )
    dev_loader = DataLoader(
        dev_data, batch_size=hyper_params["batch_size"], shuffle=False
    )
    from IPython import embed

    embed()

    return train_loader, dev_loader
