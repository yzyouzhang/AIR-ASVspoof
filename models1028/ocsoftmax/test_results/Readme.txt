Just ran test.py only for few minutes\
In case you get torch.cuda.OutOfMemoryError, just reduce batch_size from 32.\
The output will be stored in checkpoint_cm_score.txt

The terminal output after running test.py:

D:\Programming\Python\Python\venv39\Scripts\python.exe D:\Programming\Python\Python\AIR-ASVspoof-Suchit\test.py
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'resnet.ResNet' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'torch.nn.modules.conv.Conv2d' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'torch.nn.modules.batchnorm.BatchNorm2d' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'torch.nn.modules.activation.ReLU' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'torch.nn.modules.container.Sequential' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'torch.nn.modules.linear.Linear' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'loss.OCSoftmax' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
D:\Programming\Python\Python\venv39\lib\site-packages\torch\serialization.py:868: SourceChangeWarning: source code of class 'torch.nn.modules.activation.Softplus' has changed. you can retrieve the original source code by accessing the object's source attribute or set `torch.nn.Module.dump_patches = True` and use the patch tool to revert the changes.
  warnings.warn(msg, SourceChangeWarning)
  0%|          | 0/7124 [00:00<?, ?it/s]D:\Programming\Python\Python\AIR-ASVspoof-Suchit\test.py:39: UserWarning: Implicit dimension choice for softmax has been deprecated. Change the call to include dim=X as an argument.
  score = F.softmax(lfcc_outputs)[:, 0]
100%|██████████| 7124/7124 [1:40:50<00:00,  1.18it/s]
t-DCF evaluation from [Nbona=7355, Nspoof=63882] trials

   tDCF_norm(s) =  2.40595 x Pmiss_cm(s) + Pfa_cm(s)


CM SYSTEM
   EER            =  2.18635 % (Equal error rate for countermeasure)

TANDEM
   min-tDCF       =  0.05582
D:\Programming\Python\Python\AIR-ASVspoof-Suchit\test.py:83: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  asv_scores = asv_data[:, 2].astype(np.float)
D:\Programming\Python\Python\AIR-ASVspoof-Suchit\test.py:90: DeprecationWarning: `np.float` is a deprecated alias for the builtin `float`. To silence this warning, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
  cm_scores = cm_data[:, 3].astype(np.float)
t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.84359 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.89288 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  7.69295 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.90612 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.89927 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.88771 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.86927 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.83494 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.84095 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  1.84322 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) = 26.21882 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  8.85143 x Pmiss_cm(s) + Pfa_cm(s)

t-DCF evaluation from [Nbona=7355, Nspoof=4914] trials

   tDCF_norm(s) =  3.27974 x Pmiss_cm(s) + Pfa_cm(s)

[0.0013920603655477892, 0.0017995034650371156, 0.0012223292984679798, 0.011408406785701142, 0.0012223292984679798, 0.004651605161462402, 0.0022069465645264425, 0.006926532691318552, 0.014022796449716912, 0.0032595447959146125, 0.09201415951925809, 0.008963748188765185, 0.008726036156355668]
[0.00368042934276665, 0.004213527561415477, 0.009413529841999804, 0.02444090800063719, 0.00252755857420723, 0.010671671681910126, 0.006050743763236448, 0.018664248410511064, 0.037571774711576654, 0.007375005822600716, 0.6157704885614934, 0.03779138388863025, 0.031665916386706655]

Process finished with exit code 0
