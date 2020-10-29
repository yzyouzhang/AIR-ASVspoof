# AIR-ASVspoof

This repository contains our implementation of the paper submitted to ICASSP 2021, "One-class Learning towards Generalized Voice Spoofing Detection".

## Requirements
python==3.6

pytorch==1.1.0

## Data Preparation
The LFCC features are extracted with the MATLAB implementation provided by the ASVspoof 2019 organizers. Please first run the `process_LA_data.m` with MATLAB, and then run `python3 reload_data.py` with python.
You can also download our preprocessed data here.

## Run the training code
```
python3 train.py --add_loss ocsoftmax -o ./models/ocsoftmax
```
## Run the test code with trained model


