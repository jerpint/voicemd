import os
import shutil

from voicemd.predict import make_a_prediction
from voicemd.colab.clean_uploads import re_arrange_files

re_arrange_files()

for audio_file in os.listdir('./audio_files/'):
    if audio_file.endswith('wav') or audio_file.endswith('mp3'):

        make_a_prediction('./audio_files/'+audio_file, config_filepath ='./voicemd/voicemd/config.yaml',
                          best_model_path='./voicemd/model.pt')

    else:
        print(f'{audio_file} seems to have the wrong extension. We only support .wav and .mp3 work at this time.\n')
shutil.rmtree('./audio_files/')