import os

import numpy
from torch.utils.data import Dataset, DataLoader
# __TODO__ change the dataloader to suit your needs...


def get_data(data_folder, prefix):
    inputs = []
    with open(os.path.join(data_folder, '{}.input'.format(prefix))) as in_stream:
        for line in in_stream:
            inputs.append([float(x) for x in line.split()])
    in_data = numpy.array(inputs, dtype=numpy.float32)
    targets = []
    with open(os.path.join(data_folder, '{}.target'.format(prefix))) as in_stream:
        for line in in_stream:
            targets.append(float(line))
    tar_data = numpy.array(targets, dtype=numpy.float32)
    return in_data, tar_data


class MyDataset(Dataset):

    def __init__(self, in_data, tar_data):
        self.in_data = in_data
        self.tar_data = tar_data

    def __len__(self):
        return len(self.in_data)

    def __getitem__(self, index):
        tar_data = self.tar_data[index]
        data_val = self.in_data[index]
        return data_val, tar_data


def load_data(args, hyper_params):
    # __TODO__ load the data
    train_input, train_target = get_data(args.data, 'train')
    train_data = MyDataset(train_input, train_target)
    dev_input, dev_target = get_data(args.data, 'dev')
    dev_data = MyDataset(dev_input, dev_target)
    train_loader = DataLoader(train_data, batch_size=hyper_params['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=hyper_params['batch_size'], shuffle=False)
    return train_loader, dev_loader
