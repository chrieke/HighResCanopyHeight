# WHAT & THEORY
- Vegetation structure mapping, tree canopy height
- highres tree canopy height map for california and Sao Paulo
- previous products: Sentinel/Gedi based at 10m


- The estimation of canopy height from high resolution optical imagery shares similarities with the computer vision task of monocular depth
estimation.
- Vision transformers, which are a deep learning approach to
encoding low-dimensional input into a high dimensional feature space,
have established new frontiers in depth estimation compared to convolutional neural networks
- In contrast to convolutional neural networks (CNNs), which
subsequently apply local convolutional operations to enable the
modeling of increasingly long-range spatial dependencies, transformers
utilize self-attention modules to enable the modeling of global spatial
dependencies across the entire image input
- For dense prediction tasks on high resolution imagery where the
context can be sparse, such as ground information in the case of near
closed canopies, the ability of transformers to model global information
is promising.




# DATA & PREPROCESSING
- Maxar 2017-2020, around 0.5m GSD (webmercator, bing tiling scheme, zoom lvl 15, 2048x2048px, ox size 0.59m at equator). and 1m GSD aerial lidar

- Get input data:
- lat, lon
- sun elevation
- view off-Nador  (angle?)
- Terrain


# MODEL

Our method consists of an image encoder-decoder model, where low spectral
dimensional input images are transformed to a high dimensional
encoding (Phase 1) and subsequently decoded to predict per-pixel canopy height (Phase 2).


## Phase 1: ENCODER: Self supervised (SSL) architecture and training

ENCODER
- dense vision transformer decoder, self supervised training
- DINOv2 self-supervised learning > generates encodings from images
- Training: 18million 256x256 sat tiles. No labels.


## Phase 2: DECODER: Supervised archictecure and training

- Freeze the SSL encoder layers from phase 1 with the weight of the tracher model.
- Phase model generates canopy height predictions from these features, using a decoder network (trained on the features)
- Training: Linked data:
	- CHM images based on aerial Lidar (ALS, Aerial Laser Scanning)
	- RGB highres satellite image
- at training time augmentation:
	- random 90 degree rotation, brightness, contrast juttering.



# Phase 3: POSTPROCESSING, GEDI correction
- Mitigate effect of limited geographic distribution of data
- Simple CNN, regression trained on GEDI, rescales the result
- Trained on RGB Maxar images & 13million GEDI measurements (randmly sampled).
- Outputs a single value correction factor




# EVALUATION
- Evaluation with lidar, field collected data and other RS products
- MAE (mean absolute error) 2.8m, ME (mean error) 0.6



# OTHER NOTES

- GEDI CHMO is 25m res, Icesat-2 13x100m
- mostly use 95th percentile of the height measurements


- Liu et al. (2023) computed a canopy height map (CHM) map of Europe using 3 m Planet imagery, training two UNets to predict tree extent and CHM using lidar
observations and previous CHM predictions from the literature.
https://www.science.org/doi/10.1126/sciadv.adh4097
3m chm data: https://zenodo.org/records/8154445
in gee: https://ee-chm-eu-2019.projects.earthengine.app/view/euchm
code: https://zenodo.org/records/8156190

