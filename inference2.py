# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

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
    root_dir = Path('./drive/MyDrive/meta-tree-height/data/images/')
    df_path = './drive/MyDrive/meta-tree-height/data/neon_test_data.csv'
    # TODO: Remove all these
    src_img = 'neon'
    new_norm = True
    no_norm = False
    size_multiplier = 6 # number of times crops can be used horizontally
    trained_rgb = False

    def __init__(self, model_norm):
        self.model_norm = model_norm
        self.chip_size = 256 #TODO
        self.df = pd.read_csv(self.df_path, index_col=0)

    def __len__(self):
        if self.src_img == 'neon':
            return 30 * len(self.df)
        return len(self.df)

    def __getitem__(self, i):
        n = self.size_multiplier
        ix, jx, jy = i // (n ** 2), (i % (n ** 2)) // n, (i % (n ** 2)) % n
        if self.src_img == 'neon':
            l = self.df.iloc[ix]
        x = list(range(l.bord_x, l.imsize - l.bord_x - self.chip_size, self.chip_size))[
            jx]
        y = list(range(l.bord_y, l.imsize - l.bord_y - self.chip_size, self.chip_size))[
            jy]
        img = TF.to_tensor(Image.open(self.root_dir / l[self.src_img]).crop(
            (x, y, x + self.chip_size, y + self.chip_size)))
        chm = TF.to_tensor(Image.open(self.root_dir / l.chm).crop(
            (x, y, x + self.chip_size, y + self.chip_size)))
        chm[chm < 0] = 0

        if not self.trained_rgb:
            if self.src_img == 'neon':
                if self.no_norm:
                    normIn = img
                else:
                    if self.new_norm:
                        # image image normalization using learned quantiles of pairs of Maxar/Neon images
                        x = torch.unsqueeze(img, dim=0)
                        norm_img = self.model_norm(x).detach()
                        p5I = [norm_img[0][0].item(), norm_img[0][1].item(),
                               norm_img[0][2].item()]
                        p95I = [norm_img[0][3].item(), norm_img[0][4].item(),
                                norm_img[0][5].item()]
                    else:
                        # apply image normalization to aerial images, matching color intensity of maxar images
                        I = TF.to_tensor(
                            Image.open(self.root_dir / l['maxar']).crop(
                                (x, y, x + s, y + s)))
                        p5I = [np.percentile(I[i, :, :].flatten(), 5) for i in
                               range(3)]
                        p95I = [np.percentile(I[i, :, :].flatten(), 95) for i in
                                range(3)]
                    p5In = [np.percentile(img[i, :, :].flatten(), 5) for i in
                            range(3)]

                    p95In = [np.percentile(img[i, :, :].flatten(), 95) for i in
                             range(3)]
                    normIn = img.clone()
                    for i in range(3):
                        normIn[i, :, :] = (img[i, :, :] - p5In[i]) * (
                                    (p95I[i] - p5I[i]) / (p95In[i] - p5In[i])) + \
                                          p5I[i]

        return {'img': normIn}


if __name__ == '__main__':
    output_dir = Path.cwd() / "output_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda:0'

    # 1- load SSL model
    ssl_path = './drive/MyDrive/meta-tree-height/saved_checkpoints/compressed_SSLlarge.pth'
    model = SSLModule(ssl_path=ssl_path)
    model.to(device)
    model = model.eval()

    # 2- image normalization for each image going through the encoder
    image_normalizer = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    image_normalizer = image_normalizer.to(device)

    # 3- Load model to normalize aerial images to match intensities from satellite images.
    norm_path = './drive/MyDrive/meta-tree-height/saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt'
    ckpt = torch.load(norm_path, map_location='cpu')
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        if 'backbone.' in k:
            new_k = k.replace('backbone.', '')
            state_dict[new_k] = state_dict.pop(k)
    model_norm = RNet(n_classes=6)
    model_norm = model_norm.eval()
    model_norm.load_state_dict(state_dict)

    # Run prediction
    dataset = NeonDataset(model_norm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, #TODO: shuffe=True
                                             num_workers=10)

    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items() if
                 isinstance(v, torch.Tensor)}
        normalized_image = image_normalizer(batch['img'])
        pred = model(normalized_image)
        pred = pred.cpu().detach().relu()

        idx = 0
        pred_chm_array = pred[idx][0].detach().numpy()

        # Save the prediction as png image in folder
        plt.imsave(f'output_inference/pred_{idx}.png', pred_chm_array, cmap='viridis')