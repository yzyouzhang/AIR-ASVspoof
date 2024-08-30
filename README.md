AIR-ASVspoof
===============
# [One-Class Learning Towards Synthetic Voice Spoofing Detection](https://ieeexplore.ieee.org/document/9417604)

[![GitHub](https://img.shields.io/github/stars/yzyouzhang/AIR-ASVspoof)](https://github.com/yzyouzhang/AIR-ASVspoof) | [![IEEE Xplore](https://img.shields.io/badge/IEEE-9417604-E4A42C.svg)](https://ieeexplore.ieee.org/document/9417604) | [![arXiv](https://img.shields.io/badge/arXiv-2010.13995-b31b1b.svg)](https://arxiv.org/abs/2010.13995) 

This repository contains the official implementation of our SPL paper, "One-class Learning Towards Synthetic Voice Spoofing Detection."

[[poster](https://labsites.rochester.edu/air/publications/ICASSP2022Poster_Neil.pdf)]  [[slides](https://labsites.rochester.edu/air/publications/ICASSP2022Slides_Neil.pdf)]  [[video](https://www.youtube.com/watch?v=pX9aq8CaIvk)]  [[Project webpage](https://labsites.rochester.edu/air/projects/asvspoof.html)] 

[![Video](https://img.youtube.com/vi/pX9aq8CaIvk/hqdefault.jpg)](https://youtu.be/pX9aq8CaIvk)


## Updates
[Jun. 2023] We further improved the loss function by proposing SAMO algorithm (Speaker Attractor Multi-Center One-Class Learning) @ ICASSP 2023 ([Ding et al. 2023](https://ieeexplore.ieee.org/document/10094704)). [![GitHub](https://img.shields.io/github/stars/sivannavis/samo)](https://github.com/sivannavis/samo)

[Feb. 2023] We investigated one-class learning more and included new loss functions. Check out the book chapter published in *Handbook of Biometric Anti-Spoofing* ([Zhang et al. 2023](https://link.springer.com/chapter/10.1007/978-981-19-5288-3_15)). [![GitHub](https://img.shields.io/github/stars/yzyouzhang/HBAS_chapter_voice3)](https://github.com/yzyouzhang/HBAS_chapter_voice3)

[Sep. 2021] This version of the code used LFCC+ResNet as the backbone. The LFCC feature was implemented with MATLAB, and ResNet was implemented with PyTorch. If you would like full Python code, please check out our follow-up work @ Interspeech 2021 ([Zhang et al. 2021](https://www.isca-archive.org/interspeech_2021/zhang21ea_interspeech.pdf)). [![GitHub](https://img.shields.io/github/stars/yzyouzhang/Empirical-Channel-CM)](https://github.com/yzyouzhang/Empirical-Channel-CM)

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

[3] Zhang, Y., Jiang, F., Zhu, G., Chen, X., & Duan, Z. (2023). Generalizing Voice Presentation Attack Detection to Unseen Synthetic Attacks and Channel Variation. In Handbook of Biometric Anti-Spoofing: Presentation Attack Detection and Vulnerability Assessment (pp. 421-443). [[link](https://link.springer.com/chapter/10.1007/978-981-19-5288-3_15)] [[code](https://github.com/yzyouzhang/HBAS_chapter_voice3)]

[4] Ding, S., Zhang, Y., & Duan, Z. (2023). Samo: Speaker attractor multi-center one-class learning for voice anti-spoofing. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). [[link](https://ieeexplore.ieee.org/document/10094704)] [[arXiv](https://arxiv.org/abs/2211.02718)] [[code](https://github.com/sivannavis/samo)] [[video](https://youtu.be/2szWD06keUg)]
