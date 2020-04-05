# import wandb
# api = wandb.Api()
# run = api.run("alx/voicemd/runs/2kke7hm9")
#
# run.config["batch_size"] = 16
# run.config["optimizer"] = 'sgd'
# run.config["scheduler"] = 'CyclicLR_2'
# run.config["architecture"] = 'simplecnn'
# run.config["in_channels"] = 1
# run.config["pretrained"] = True
# run.config["normalize_spectrums"] = True
# run.config["learning_rate"] = 0.01
# run.config["window_len"] = 256
# run.config["spec_type"] = 'librosa_melspec'
# run.config["split_type"] = 'rand_shuffle'
# run.config["split_rand_state"] = 42
# run.config["base_lr"] = 0.001
# run.config["max_lr"] = 0.01
# run.update()
#
# # architecture
# size: 10
