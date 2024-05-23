# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from pathlib import Path
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import torchvision.transforms.functional as TF
import rasterio
import rasterio.windows
from rasterio.crs import CRS

from models.backbone import SSLVisionTransformer
from models.dpt_head import DPTHead
import pytorch_lightning as pl
from models.regressor import RNet


class SSLAE(nn.Module):
    def __init__(self, pretrained=None, classify=True, n_bins=256, huge=False):
        super().__init__()
        if huge == True:
            self.backbone = SSLVisionTransformer(
                embed_dim=1280,
                num_heads=20,
                out_indices=(9, 16, 22, 29),
                depth=32,
                pretrained=pretrained
            )
            self.decode_head = DPTHead(
                classify=classify,
                in_channels=(1280, 1280, 1280, 1280),
                embed_dims=1280,
                post_process_channels=[160, 320, 640, 1280],
            )
        else:
            self.backbone = SSLVisionTransformer(pretrained=pretrained)
            self.decode_head = DPTHead(classify=classify, n_bins=256)

    def forward(self, x):
        x = self.backbone(x)
        x = self.decode_head(x)
        return x


class SSLModule(pl.LightningModule):
    def __init__(self,
                 ssl_path="compressed_SSLbaseline.pth"):
        super().__init__()

        if 'huge' in ssl_path:
            self.chm_module_ = SSLAE(classify=True, huge=True).eval()
        else:
            self.chm_module_ = SSLAE(classify=True, huge=False).eval()

        if 'compressed' in ssl_path:
            ckpt = torch.load(ssl_path, map_location='cpu')
            self.chm_module_ = torch.quantization.quantize_dynamic(
                self.chm_module_,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d},
                dtype=torch.qint8)
            self.chm_module_.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(ssl_path)
            state_dict = ckpt['state_dict']
            self.chm_module_.load_state_dict(state_dict)

        self.chm_module = lambda x: 10 * self.chm_module_(x)

    def forward(self, x):
        x = self.chm_module(x)
        return x


class NeonDataset(torch.utils.data.Dataset):
    # # TODO: Remove all these
    # src_img = 'neon'
    # new_norm = True
    # no_norm = False
    # size_multiplier = 6 # number of times crops can be used horizontally
    # trained_rgb = None

    def __init__(self, base_dir, model_norm):
        self.root_dir = Path(base_dir) / "data/images/"
        self.image_paths = list(self.root_dir.glob("*.TIF"))
        self.model_norm = model_norm
        self.chip_size = 256 #TODO
        #self.df_path = base_dir / "data/neon_test_data.csv"
        #self.df = pd.read_csv(self.df_path, index_col=0)

    def __len__(self): #TODO
        length = len(self.image_paths)
        if not length:
            print("DATASET LENGTH IS 0!!!!!!!!!!!!!")
        return length

    def __getitem__(self, i):
        image_fp = self.image_paths[i]

        # TODO iterative, maybe another library
        col_off, row_off, width, height = 0, 0, self.chip_size, self.chip_size
        window = rasterio.windows.Window(col_off, row_off, width, height)

        with rasterio.open(image_fp) as src:
            clipped_img = src.read([1,2,3], window=window)
            clipped_transform = rasterio.windows.transform(window, src.transform)
            profile = src.profile.copy()
            profile.update(
                {
                    "transform": clipped_transform,
                    "width": clipped_img.shape[1],
                    "height": clipped_img.shape[2],
                    "count": 1,
                    "crs": str(src.crs.to_epsg())
                }
            )
            profile.pop("nodata", None)

        img = TF.to_tensor(np.moveaxis(clipped_img, 0, -1)) # Change from rasterios # Shape (3, 256, 256) to (256, 256, 3), torch expects that

        # image normalization using learned quantiles of pairs of Maxar/Neon images
        x = torch.unsqueeze(img, dim=0)
        norm_img = self.model_norm(x).detach()
        p5I = [norm_img[0][0].item(), norm_img[0][1].item(),
               norm_img[0][2].item()]
        p95I = [norm_img[0][3].item(), norm_img[0][4].item(),
                norm_img[0][5].item()]
        p5In = [np.percentile(img[i, :, :].flatten(), 5) for i in
                range(3)]
        p95In = [np.percentile(img[i, :, :].flatten(), 95) for i in
                 range(3)]
        normalized_img = img.clone()
        for i in range(3):
            normalized_img[i, :, :] = (img[i, :, :] - p5In[i]) * (
                        (p95I[i] - p5I[i]) / (p95In[i] - p5In[i])) + \
                              p5I[i]

        return normalized_img, profile # batch must contain tensors, numpy arrays, numbers, dicts or lists, no rasterio, no nodata etc.


if __name__ == '__main__':
    #base_dir = Path.cwd()
    base_dir = Path('./drive/MyDrive/meta-tree-height')

    output_dir = base_dir / "output_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cpu" #'cuda:0' #TODO search cpu, other locations! if 'compressed' in args.checkpoint:

    # 1- load SSL model
    ssl_path = base_dir / 'saved_checkpoints/compressed_SSLlarge.pth'
    model = SSLModule(ssl_path=str(ssl_path))
    model.to(device)
    model = model.eval()

    # 2- image normalization for each image going through the encoder
    image_normalizer = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    image_normalizer = image_normalizer.to(device)

    # 3- Load model to normalize aerial images to match intensities from satellite images.
    norm_path = base_dir / 'saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt'
    ckpt = torch.load(str(norm_path), map_location='cpu')
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        if 'backbone.' in k:
            new_k = k.replace('backbone.', '')
            state_dict[new_k] = state_dict.pop(k)
    model_norm = RNet(n_classes=6)
    model_norm = model_norm.eval()
    model_norm.load_state_dict(state_dict)

    # TODO: Here with dataset loader?
    # Run prediction
    dataset = NeonDataset(base_dir, model_norm)
    normalized_image, profile = dataset.__getitem__(0)

    # Can maybe be removed
    out_profile = copy.deepcopy(profile)
    out_profile["crs"] = CRS.from_epsg(out_profile['crs'])
    normalized_image = normalized_image.to(device)
    normalized_image = image_normalizer(normalized_image)
    print(normalized_image.shape)

    normalized_image = normalized_image.unsqueeze(0) #adds batch dimension (1,3,256,256) instead of (3,256,256)
    pred = model(normalized_image)
    pred = pred.cpu().detach().relu()

    for idx in range(pred.shape[0]): #is this only  asingle image always?
        pred_chm_array = pred[idx][0].detach().numpy()

        # Export
        plt.imsave(f'output_inference/AAApred_{idx}.png', pred_chm_array, cmap='viridis')
        geotiff_file_out = base_dir / f'output_inference/pred_{idx}.tif'
        # Save the prediction as png image in folder
        with rasterio.open(geotiff_file_out, 'w', **out_profile) as dst:
            dst.write(pred_chm_array, 1)

        print("FINISHED!, exported to", geotiff_file_out)
