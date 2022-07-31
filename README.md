# Fusing Sentinel-2 and Worldview imagery

This repository contains code related to the implementation of the 3 deep learning CNNs (Fusion-PNN-Siamese, Fusion-ResNet, Fusion-GAN) which were created in order to fuse Sentinel-2 (20 m) and Worldview images (4 m), as analyzed in the paper cited below.

Kremezi, M., Kristollari, V., Karathanassi, V., Topouzelis, K., Kolokoussis, P., Taggio, N., Aiello, A., Ceriola, G., Barbone, E. and Corradi, P., 2022. Increasing the Sentinel-2 potential for marine plastic litter monitoring through image fusion techniques. Marine Pollution Bulletin, 182, 113974.

It can be accessed in: https://www.sciencedirect.com/science/article/pii/S0025326X22006567#f0105.
The preprint can be accessed in: http://users.ntua.gr/vkristoll/assets/MBP_paper_preprint.pdf

![DL fusion](/images/DL_fusion.PNG)
![DL models](/images/models.png)


## Steps to implement the code
Run:
>1. "training_preprocessing.py" to apply pre-processing to the training input and output files.
>
>2. "training.py" to train each model.
>
>3. "inference_preprocessing.py" to apply pre-processing to the inference input files.
>
>4. "inference.py" to create the fused image.
>
>5. "remove_01_scaling.py" to create uint16 values.
>
>6. "remove_histogram_clipping.py" to remove the histogram clipping effect.

*Detailed guidelines are included inside each script.*


If you use this code, please cite the below paper.

```
@article{kremezi2022fusion,
author = {Maria Kremezi and Viktoria Kristollari and Vassilia Karathanassi and Konstantinos Topouzelis and Pol Kolokoussis and Nicolò Taggio and Antonello Aiello and Giulio Ceriola and Enrico Barbone and Paolo Corradi},
title = {Increasing the Sentinel-2 potential for marine plastic litter monitoring through image fusion techniques},
journal = {Marine Pollution Bulletin},
volume = {182},
pages = {113974},
year = {2022},
doi = {https://doi.org/10.1016/j.marpolbul.2022.113974}
}
```
