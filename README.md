AIR-ASVspoof
===============
This repository contains our implementation of the paper, "One-class Learning Towards Synthetic Voice Spoofing Detection".
[Paper link here](https://arxiv.org/pdf/2010.13995.pdf)
## Requirements
python==3.6

pytorch==1.1.0

## Data Preparation
The LFCC features are extracted with the MATLAB implementation provided by the ASVspoof 2019 organizers. Please first run the `process_LA_data.m` with MATLAB, and then run `python3 reload_data.py` with python.
Make sure you change the directory path to the path on your machine.
## Run the training code
Before running the `train.py`, please change the `path_to_database`, `path_to_features`, `path_to_protocol` according to the files' location on your machine.
```
python3 train.py --add_loss ocsoftmax -o ./models/ocsoftmax --gpu 0
```
## Run the test code with trained model
You can change the `model_dir` to the location of the model you would like to test with.
```
python3 test.py -m ./models/ocsoftmax -l ocsoftmax --gpu 0
```

