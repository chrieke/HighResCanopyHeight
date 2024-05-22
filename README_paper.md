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
is promising




# DATA
- Maxar 2017-2020, around 0.5m GSD (webmercator, bing tiling scheme, zoom lvl 15, 2048x2048px, ox size 0.59m at equator). and 1m GSD aerial lidar



# DATA PREP




# MODEL / METHODOLOGY

- Dense prediction decoders against aerial lidar maps
- Our method consists of an image encoder-decoder model, where low spectral
dimensional input images are transformed to a high dimensional
encoding and subsequently decoded to predict per-pixel canopy height
- We employ DINOv2 self-supervised learning to generate universal and
generalizable encodings from the input imagery (Oquab et al., 2023),
and train a dense vision transformer decoder (Ranftl et al., 2021) to
generate canopy height predictions based on aerial lidar data from sites
across the USA




# POSTPROCESSING
- postprocessing CNN trained on GEDI
- To correct a potential bias coming from a geographically
limited source of supervision, we finally refine the maps using a convolutional network trained on spaceborne lidar data





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







Product question:
- How would we surface 1m CHM?
- How would we surface or make use if the additional 3m/10m chms?




Long term ideas:
	- retrain using the Sweden Lidar data