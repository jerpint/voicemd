import torch
import yaml
import numpy as np
from tqdm import tqdm
from yaml import load
from voicemd.data.prepare_dataloaders import make_predict_dataloader
from voicemd.models.model_loader import load_model


def make_a_prediction(sound_filepath, config_filepath ='/voicemd/config.yaml',
                      best_model_path='voicemd/output/best_model.pt'):

    sound_filename = sound_filepath.split('/')[-1]
    print(f'Analyzing {sound_filename}...')

    with open(config_filepath, 'r') as stream:
        hyper_params = load(stream, Loader=yaml.FullLoader)
    model = load_model(hyper_params)
    model.load_state_dict(torch.load(best_model_path))
    predict_dataloader = make_predict_dataloader(sound_filepath, hyper_params)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    all_probs = []

    model.to(device)

    for data in tqdm(predict_dataloader):
        pred = model(data.to(device))
        probs = torch.nn.functional.softmax(pred, dim=1)
        all_probs.extend(probs.detach().cpu().numpy())

    all_probs = np.array(all_probs)
    avg_prob = np.sum(all_probs, 0) / len(all_probs)

    print(f"{sound_filename} probability to be a male's voice: {round((avg_prob[1]*100),2)}%")
    print(f"{sound_filename} probability to be a female's voice: {round((avg_prob[0]*100),2)}%\n")
