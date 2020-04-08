import torch
import yaml
from yaml import load
import numpy as np

from voicemd.data.prepare_dataloaders import make_predict_dataloader
from voicemd.models.model_loader import load_model

# Adapt these filepaths to suit your needs
config_filename = '/home/jerpint/voicemd/voicemd/config.yaml'
sound_filename = '/home/jerpint/voicemd/data/voice_clips/BL03.wav'
best_model_path = '/home/jerpint/voicemd/voicemd/simple_cnn_best/best_model'

with open(config_filename, 'r') as stream:
    hyper_params = load(stream, Loader=yaml.FullLoader)
model = load_model(hyper_params)
model.load_state_dict(torch.load(best_model_path))
predict_dataloader = make_predict_dataloader(sound_filename, hyper_params)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

all_probs = []
for data in predict_dataloader:

    pred = model(data.to(device))
    probs = torch.nn.functional.softmax(pred, dim=1)
    all_probs.extend(probs.detach().numpy())

all_probs = np.array(all_probs)
avg_prob = np.sum(all_probs, 0) / len(all_probs)
print("Confidence prediction of male %:", avg_prob[1]*100)
print("Confidence prediction of female %:", avg_prob[0]*100)
