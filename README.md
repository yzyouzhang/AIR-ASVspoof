AIR-ASVspoof
===============
This repository contains the official implementation of "One-class Learning Towards Synthetic Voice Spoofing Detection".
[[paper](https://ieeexplore.ieee.org/document/9417604)] [[arXiv](https://arxiv.org/pdf/2010.13995.pdf)] [[video](https://www.youtube.com/watch?v=pX9aq8CaIvk)]\
You can find a brief explanation [here.](https://suchitreddi.github.io/Work/Voice_Spoofing/)
## Requirements
Install python 3.9 and the latest version of pytorch (with/without CUDA, depending on your device).\
If you encounter any problems with python, try installing python==3.6 and pytorch==1.1.0. (Required for Training of data)\
Note:- Pytorch is currently compatible with only python versions from 3.7 to 3.9. So, that might cause some errors in torch packages used for `train.py` and others.\
But, you can run the testing part with the already provided pre-trained model.\
Check this issue for more information: https://github.com/yzyouzhang/AIR-ASVspoof/issues/36
 (The inputs of test should be `anti-spoofing_lfcc_model.pt` and `anti-spoofing_loss_model.pt`, output should be `checkpoint_cm_score.txt`)

## Downloads
If you want to work on this project, you will have to download a few things.
1. An IDE(IntelliJ) to run .py files and MATLAB to run .m files (`feature_extraction.py` is the python version of `process_LA_data.m` file.).
2. After installing python, download CUDA if you have supporting NVIDIA gpu (for faster testing and training), and download pytorch with/without CUDA accordingly from [here](https://pytorch.org/get-started/previous-versions/).
3. For Data Preparation, database of audio samples (LA.zip) should be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336).\
Extract LA.zip into DS_10283_3336(Dataset) folder.\
MATLAB implementations for LFCC features are available [here](https://www.asvspoof.org/index2019.html). 
## Data Preparation
The LFCC features are extracted with the MATLAB implementation provided by the ASVspoof 2019 organizers.\
First run the `process_LA_data.m` with MATLAB, and then run python `reload_data.py` with python.\
If you don't have/want to use MATLAB, use the python file `feature_extraction.py`, instead of `process_LA_data.m`, then run `reload_data.py`.\
Make sure you change the directory path to the path on your machine.

For successful execution of `process_LA_data.m`, we will have to create a folder to contain the output .m files.\
Create a folder `anti-spoofing` inside DS_10283_3336 as shown in pathToFeatures line of the `process_LA_data.m` file.\
The output folder there in `process_LA_data.m` is the Features folder. It should be further divided into dev, eval and train folders.
## Run the training code
Before running the `train.py`, please change the `path_to_database`, `path_to_features`, `path_to_protocol` according to the files location on your machine.\
`path_to_database` is the path to your DS_10283_3336 folder.\
`path_to_features` is the path to the output folder created for **Data Preperation** part above.\
`path_to_protocol` is the path to ASVspoof2019_LA_cm_protocols folder.
```
python train.py --add_loss ocsoftmax -o D:/Programming/Python/Python/AIR-ASVspoof-Suchit/models1028/ocsoftmax/test_results --gpu 0
```
Change --gpu 0 to --gpu 1, if you want to run the code with your nvidia gpu. 
## Run the test code with trained model
You can change the `model_dir` to the location of the trained model you would like to test with.\
Make sure that you change all the required paths in the code to your folder paths.
```
python test.py -m D:/Programming/Python/Python/AIR-ASVspoof-Suchit/models1028/ocsoftmax/test_results -l ocsoftmax
```
The output will be inside the text file `checkpoint_cm_score.txt` in `model_dir` folder.\
The output will be in the form of: `LA_E_2834763 A11 spoof -0.9655166864395142`.

The final output of the model with a min t-DCF of 0.059, and EER of 2.19%. This model could be in the 3rd position of ASVspoof 2019 competition's LA subset.

You can find citations and future works in the author's repository [here.](https://github.com/yzyouzhang/AIR-ASVspoof)
