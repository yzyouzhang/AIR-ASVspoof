AIR-ASVspoof
===============
This repository contains the official implementation of our SPL paper, "One-class Learning Towards Synthetic Voice Spoofing Detection".
[[Paper link here](https://ieeexplore.ieee.org/document/9417604)] [[arXiv](https://arxiv.org/pdf/2010.13995.pdf)] [[video](https://www.youtube.com/watch?v=pX9aq8CaIvk)]
## Requirements
python==3.6 and pytorch==1.1.0.
Note:- Pytorch is currently compatible with only python versions from 3.7 to 3.9. So, that might cause some errors in torch packages used for `train.py` and others.

## Downloads
If you want to work on this project. You will have to download a few things.
1. A python supporting IDE to run .py files and MATLAB to run .m files (`feature_extraction.py` is the python version of `process_LA_data.m` file).
2. After installing python 3.6. Be sure to download CUDA if you have supporting NVIDIA gpu, and download pytorch with or without CUDA accordingly from [here](https://pytorch.org/get-started/previous-versions/).(Search for 1.1.0 version).
3. For Data Preparation; Database of audio samples (LA.zip and PA.zip) should be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/3336). Extract LA and PA zips and place them in DS_10283_3336 folder.
   MATLAB implementations for LFCC features are available [here](https://www.asvspoof.org/index2019.html). 
## Data Preparation
The LFCC features are extracted with the MATLAB implementation provided by the ASVspoof 2019 organizers. Please first run the `process_LA_data.m` with MATLAB, and then run python `reload_data.py` with python.
If you don't have or don't want to use MATLAB, use the python file `feature_extraction.py`, instead of `process_LA_data.m`, and then run `reload_data.py`.
Make sure you change the directory path to the path on your machine.

For successful execution of `process_LA_data.m`, we will have to create a folder to contain the output .m files. Create a folder `anti-spoofing` inside DS_10283_3336 as shown in pathToFeatures line of the `process_LA_data.m` file. The output folder there is the Features folder. It should be further divided into dev, eval and train folders.
## Run the training code
Before running the `train.py`, please change the `path_to_database`, `path_to_features`, `path_to_protocol` according to the files location on your machine.
`path_to_database` is the path to your DS_10283_3336 folder. `path_to_features` is the path to the output folder created for **Data Preperation** part above. `path_to_protocol` is the path to ASVspoof2019_LA_cm_protocols folder.
```
python train.py --add_loss ocsoftmax -o D:/Programming/Python/Python/AIR-ASVspoof-Suchit/models1028/ocsoftmax/test_results --gpu 0
```
Change --gpu 0 to --gpu 1 if you want to run the code with your nvidia gpu. 
## Run the test code with trained model
You can change the `model_dir` to the location of the trained model you would like to test with. 
Make sure that you change all the required paths in the code to your folder paths.
```
python test.py -m D:/Programming/Python/Python/AIR-ASVspoof-Suchit/models1028/ocsoftmax/test_results -l ocsoftmax
```
The output will be inside the text file `checkpoint_cm_score.txt` in `model_dir` folder. The output will be in the form of:
`LA_E_2834763 A11 spoof -0.9655166864395142`.
## Citation
```
@ARTICLE{zhang2021one,
  author={Zhang, You and Jiang, Fei and Duan, Zhiyao},
  journal={IEEE Signal Processing Letters}, 
  title={One-Class Learning Towards Synthetic Voice Spoofing Detection}, 
  year={2021},
  volume={28},
  number={},
  pages={937-941},
  abstract={Human voices can be used to authenticate the identity of the speaker, but the automatic speaker verification (ASV) systems are vulnerable to voice spoofing attacks, such as impersonation, replay, text-to-speech, and voice conversion. Recently, researchers developed anti-spoofing techniques to improve the reliability of ASV systems against spoofing attacks. However, most methods encounter difficulties in detecting unknown attacks in practical use, which often have different statistical distributions from known attacks. Especially, the fast development of synthetic voice spoofing algorithms is generating increasingly powerful attacks, putting the ASV systems at risk of unseen attacks. In this work, we propose an anti-spoofing system to detect unknown synthetic voice spoofing attacks (i.e., text-to-speech or voice conversion) using one-class learning. The key idea is to compact the bona fide speech representation and inject an angular margin to separate the spoofing attacks in the embedding space. Without resorting to any data augmentation methods, our proposed system achieves an equal error rate (EER) of 2.19% on the evaluation set of ASVspoof 2019 Challenge logical access scenario, outperforming all existing single systems (i.e., those without model ensemble).},
  keywords={},
  doi={10.1109/LSP.2021.3076358},
  ISSN={1558-2361},
  month={},}
```

## Follow-up works
Please check out our follow-up work:

[1] Zhang, Y., Zhu, G., Jiang, F., Duan, Z. (2021) An Empirical Study on Channel Effects for Synthetic Voice Spoofing Countermeasure Systems. Proc. Interspeech 2021, 4309-4313, doi: 10.21437/Interspeech.2021-1820 [[link](https://www.isca-speech.org/archive/interspeech_2021/zhang21ea_interspeech.html)] [[arXiv](https://arxiv.org/pdf/2104.01320.pdf)] [[code](https://github.com/yzyouzhang/Empirical-Channel-CM)] [[video](https://www.youtube.com/watch?v=vLijNUJklo0)]

[2] Chen, X., Zhang, Y., Zhu, G., Duan, Z. (2021) UR Channel-Robust Synthetic Speech Detection System for ASVspoof 2021. Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge, 75-82, doi: 10.21437/ASVSPOOF.2021-12 [[link](https://www.isca-speech.org/archive/asvspoof_2021/chen21_asvspoof.html)] [[arXiv](https://arxiv.org/pdf/2107.12018.pdf)] [[code](https://github.com/yzyouzhang/ASVspoof2021_AIR)] [[video](https://www.youtube.com/watch?v=-wKMOTp8Tt0)]

