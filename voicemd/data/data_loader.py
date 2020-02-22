import os
import pandas as pd

import numpy
from torch.utils.data import Dataset, DataLoader
from voicemd.data.dataloaders import AudioDataset

# __TODO__ change the dataloader to suit your needs...


def load_data(args, hyper_params):
    # __TODO__ load the data
    # TODO: Add splits for validation
    fname = args.data + "cleaned_metadata.csv"
    metadata = pd.read_csv(fname)
    metadata = metadata[metadata["filename"].notna()]
    normalize = hyper_params['normalize_spectrums']

    random_state = 43
    if hyper_params['debug']:
        train_metadata = valid_metadata = metadata.iloc[0:10]

    else:
        # Split male and female, shuffle them, join them back and sample
        male_metadata = metadata[metadata['gender'] == 'M']
        female_metadata = metadata[metadata['gender'] == 'F']

        male_metadata_shuffle = male_metadata.sample(n=len(male_metadata), random_state=random_state)
        female_metadata_shuffle = female_metadata.sample(n=len(female_metadata), random_state=random_state)

        train_metadata = male_metadata_shuffle[0:75].append(female_metadata_shuffle[0:75])
        train_metadata = train_metadata.sample(n=len(train_metadata), random_state=random_state)

        valid_metadata = male_metadata_shuffle[75:100].append(female_metadata_shuffle[75:100])
        valid_metadata = valid_metadata.sample(n=len(valid_metadata), random_state=random_state)

    train_data = AudioDataset(
        train_metadata,
        voice_clips_dir=args.data,
        spec_type=hyper_params['spec_type'],
        window_len=hyper_params['window_len'],
        in_channels=hyper_params['in_channels'],
        normalize=normalize,

    )

    dev_data = AudioDataset(
        valid_metadata,
        voice_clips_dir=args.data,
        spec_type=hyper_params['spec_type'],
        window_len=hyper_params['window_len'],
        in_channels=hyper_params['in_channels'],
        normalize=normalize,
    )

    train_loader = DataLoader(
        train_data, batch_size=hyper_params["batch_size"], shuffle=True
    )

    dev_loader = DataLoader(
        dev_data, batch_size=hyper_params["batch_size"], shuffle=False
    )

    return train_loader, dev_loader
