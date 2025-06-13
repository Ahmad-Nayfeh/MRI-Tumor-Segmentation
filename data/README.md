# Data Directory

This directory contains the MRI brain tumor dataset used for training and evaluation.

## Dataset
We use the **BraTS-Africa** dataset from The Cancer Imaging Archive (TCIA).

**Download Link:** https://www.cancerimagingarchive.org/collection/brats-africa/

## Structure
```
data/
├── raw/          # Place downloaded .nii.gz files here
└── processed/    # Generated 2D slices (created by preprocessing notebook)
```

## Citation
When using this dataset, please cite:

Adewole, M., Rudie, J.D., Gbadamosi, A., Zhang, D., Raymond, C., Ajigbotoshso, J., Toyobo, O., Aguh, K., Omidiji, O., Akinola R., Suwaid, M.A., Emegoakor, A., Ojo, N., Kalaiwo, C., Babatunde, G., Ogunleye, A., Gbadamosi, Y., Iorpagher, K., Onuwaje M., Betiku B., Saluja, R., Menze, B., Baid, U., Bakas, S., Dako, F., Fatade A., Anazodo, U.C. (2024) Expanding the Brain Tumor Segmentation (BraTS) data to include African Populations (BraTS-Africa) (version 1) [Dataset]. The Cancer Imaging Archive. https://doi.org/10.7937/v8h6-8×67