[![Build Status](https://travis-ci.com/jerpint/voicemd.png?branch=master)](https://travis-ci.com/jerpint/voicemd)

# voicemd


Voice classification

## Setup

Clone the repo and install the project dependencies. It is recommended to use a virtual environment (e.g. conda)

    git clone https://github.com/jerpint/voicemd
    cd voicemd
    pip install -e .


## Make a prediction

### From script
First, download the weights locally:

    cd ~/voicemd/voicemd
    curl -L -o model.pt "https://www.dropbox.com/s/r3ydpfxr3h4vlzg/best_model_commonvoice?dl=1"

Then run as a script:

    from voicemd.predict import make_a_prediction

    make_a_prediction(
        sound_filepath='/path/to/your/sound.wav',
        config_filepath='config.yaml',
        best_model_path='model.pt'
    )


### Online Demo (Colab)

* Small dataset: Use the demo colab [here](https://colab.research.google.com/github/jerpint/voicemd/blob/master/voicemd/colab/VoiceMD.ipynb)
* Commonvoice dataset: Use the demo colab [here](https://colab.research.google.com/github/jerpint/voicemd/blob/master/voicemd/colab/VoiceMD-CV.ipynb)


