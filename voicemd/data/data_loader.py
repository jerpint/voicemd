import os
import pandas as pd

import numpy
from torch.utils.data import Dataset, DataLoader
from voicemd.data.dataloaders import TrainDataset, EvalDataset


def load_metadata(args, hyper_params):
    fname = args.data + "cleaned_metadata.csv"
    metadata = pd.read_csv(fname)
    metadata = metadata.drop(columns=['Unnamed: 0'])
    metadata = metadata.set_index('uid')
    metadata = metadata[metadata["filename"].notna()]
    return metadata


def get_metadata_splits(args, hyper_params):

    metadata = load_metadata(args, hyper_params)

    if hyper_params['split_type'] == 'even_split':
        # Split male and female, shuffle them,
        # join them back and sample evenly from both
        # Splitting evenly should only be for debugging
        male_metadata = metadata[metadata['gender'] == 'M']
        female_metadata = metadata[metadata['gender'] == 'F']

        male_metadata_shuffle = male_metadata.sample(n=len(male_metadata), random_state=hyper_params['split_rand_state'])
        female_metadata_shuffle = female_metadata.sample(n=len(female_metadata), random_state=hyper_params['split_rand_state'])

        train_metadata = male_metadata_shuffle[0:75].append(female_metadata_shuffle[0:75])
        train_metadata = train_metadata.sample(n=len(train_metadata), random_state=hyper_params['split_rand_state'])

        valid_metadata = male_metadata_shuffle[75:100].append(female_metadata_shuffle[75:100])
        valid_metadata = valid_metadata.sample(n=len(valid_metadata), random_state=hyper_params['split_rand_state'])

        test_metadata = male_metadata_shuffle[100:].append(female_metadata_shuffle[100:])
        test_metadata = test_metadata.sample(n=len(test_metadata), random_state=hyper_params['split_rand_state'])

    elif hyper_params['split_type'] == 'rand_shuffle':
        shuffled_metadata = metadata.sample(n=len(metadata), random_state=hyper_params['split_rand_state'])

        train_percentage = 0.8
        val_percentage = 0.1
        # test_percentage = 0.1

        train_metadata = shuffled_metadata[0:round(len(shuffled_metadata) * train_percentage)]
        valid_metadata = shuffled_metadata[round(len(shuffled_metadata) * train_percentage):round(len(shuffled_metadata)*(train_percentage + val_percentage))]
        test_metadata = shuffled_metadata[round(len(shuffled_metadata) * (train_percentage + val_percentage)):]

    elif hyper_params['split_type'] == 'debug':
        shuffled_metadata = metadata.sample(n=len(metadata), random_state=hyper_params['split_rand_state'])
        train_metadata = shuffled_metadata.iloc[0:32]
        valid_metadata = shuffled_metadata.iloc[0:32]
        test_metadata = shuffled_metadata.iloc[0:32]

    else:
        raise NotImplementedError("split_type not defined")

    return train_metadata, valid_metadata, test_metadata


def load_data(args, hyper_params):
    # TODO: Add splits for validation

    train_metadata, valid_metadata, test_metadata = get_metadata_splits(args, hyper_params)

    train_data = TrainDataset(
        train_metadata,
        voice_clips_dir=args.data,
        spec_type=hyper_params['spec_type'],
        window_len=hyper_params['window_len'],
        in_channels=hyper_params['in_channels'],
        preprocess=True,
        normalize=hyper_params['normalize_spectrums'],

    )

    valid_data = TrainDataset(
        valid_metadata,
        voice_clips_dir=args.data,
        spec_type=hyper_params['spec_type'],
        window_len=hyper_params['window_len'],
        in_channels=hyper_params['in_channels'],
        preprocess=True,
        normalize=hyper_params['normalize_spectrums'],
    )

    test_data_list = []

    test_data = EvalDataset(
        test_metadata.iloc[[0]],
        voice_clips_dir=args.data,
        spec_type=hyper_params['spec_type'],
        window_len=hyper_params['window_len'],
        in_channels=hyper_params['in_channels'],
        preprocess=True,
        normalize=hyper_params['normalize_spectrums'],
    )

    train_loader = DataLoader(
        train_data, batch_size=hyper_params["batch_size"], shuffle=True
    )

    valid_loader = DataLoader(
        valid_data, batch_size=hyper_params["batch_size"], shuffle=False
    )

    test_loader = DataLoader(
        test_data, batch_size=hyper_params["batch_size"], shuffle=False
    )

    return train_loader, valid_loader, test_loader
