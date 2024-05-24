# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from pathlib import Path

import copy
import torch

torch.backends.quantized.engine = "qnnpack"
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import rasterio
import rasterio.windows
from rasterio.crs import CRS

from models.regressor import RNet
from models.models import SSLModule


class NeonDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, model_norm, chip_size=256):
        self.image_dir = Path(image_dir)
        self.image_paths = list(image_dir.glob("*.TIF"))
        self.model_norm = model_norm
        self.chip_size = chip_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_fp = self.image_paths[idx]

        # TODO iterative, maybe torchvision
        col_off, row_off, width, height = 0, 0, self.chip_size, self.chip_size
        window = rasterio.windows.Window(col_off, row_off, width, height)

        with rasterio.open(image_fp) as src:
            clipped_img = src.read([1, 2, 3], window=window)
            clipped_transform = rasterio.windows.transform(window, src.transform)
            profile = src.profile.copy()
            profile.update(
                {
                    "transform": clipped_transform,
                    "width": clipped_img.shape[1],
                    "height": clipped_img.shape[2],
                    "count": 1,
                    "crs": str(src.crs.to_epsg()),
                }
            )
            profile.pop("nodata", None)

        img = TF.to_tensor(
            np.moveaxis(clipped_img, 0, -1)
        )  # Change from rasterios # Shape (3, 256, 256) to (256, 256, 3), torch expects that

        # image normalization using learned quantiles of pairs of Maxar/Neon images
        x = torch.unsqueeze(img, dim=0)
        norm_img = self.model_norm(x).detach()
        p5I = [norm_img[0][0].item(), norm_img[0][1].item(), norm_img[0][2].item()]
        p95I = [norm_img[0][3].item(), norm_img[0][4].item(), norm_img[0][5].item()]
        p5In = [np.percentile(img[i, :, :].flatten(), 5) for i in range(3)]
        p95In = [np.percentile(img[i, :, :].flatten(), 95) for i in range(3)]
        normalized_img = img.clone()
        for i in range(3):
            normalized_img[i, :, :] = (img[i, :, :] - p5In[i]) * (
                (p95I[i] - p5I[i]) / (p95In[i] - p5In[i])
            ) + p5I[i]

        return (
            normalized_img,
            profile,
        )  # batch must contain tensors, numpy arrays, numbers, dicts or lists, no rasterio, no nodata etc.


if __name__ == "__main__":
    base_dir = Path.cwd()
    # base_dir = Path('./drive/MyDrive/meta-tree-height')

    image_dir = base_dir / "data/images/"
    output_dir = base_dir / "output_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    ssl_model_fp = base_dir / "saved_checkpoints/compressed_SSLlarge.pth"
    normalization_model_fp = (
        base_dir / "saved_checkpoints/aerial_normalization_quantiles_predictor.ckpt"
    )

    device = "cpu"  #'cuda:0' #TODO search cpu, other locations! if 'compressed' in args.checkpoint:

    # 1- load SSL model
    model = SSLModule(ssl_path=str(ssl_model_fp))
    model.to(device)
    model = model.eval()

    # 2- image normalization for each image going through the encoder
    image_normalizer = T.Normalize((0.420, 0.411, 0.296), (0.213, 0.156, 0.143))
    image_normalizer = image_normalizer.to(device)

    # 3- Load model to normalize aerial images to match intensities from satellite images.
    ckpt = torch.load(str(normalization_model_fp), map_location="cpu")  # TODO
    state_dict = ckpt["state_dict"]
    for k in list(state_dict.keys()):
        if "backbone." in k:
            new_k = k.replace("backbone.", "")
            state_dict[new_k] = state_dict.pop(k)
    model_norm = RNet(n_classes=6)
    model_norm = model_norm.eval()
    model_norm.load_state_dict(state_dict)

    # TODO: Here with dataset loader?
    # Get data
    dataset = NeonDataset(image_dir=image_dir, model_norm=model_norm, chip_size=256)
    if not dataset.__len__():
        raise ValueError("Dataset length is 0!")

    normalized_image, profile = dataset.__getitem__(0)

    out_profile = copy.deepcopy(profile)
    out_profile["crs"] = CRS.from_epsg(out_profile["crs"])
    normalized_image = normalized_image.to(device)
    normalized_image = image_normalizer(normalized_image)

    normalized_image = normalized_image.unsqueeze(
        0
    )  # adds batch dimension (1,3,256,256) instead of (3,256,256)

    # Run prediction
    pred = model(normalized_image)
    pred = pred.cpu().detach().relu()

    # Export chips
    for idx in range(pred.shape[0]):  # is this only  asingle image always?
        pred_chm_array = pred[idx][0].detach().numpy()
        # plt.imsave(
        #     f"{output_dir}/AAApred_{idx}.png", pred_chm_array, cmap="viridis"
        # )
        geotiff_file_out = output_dir / f"output_inference/pred_{idx}.tif"
        with rasterio.open(geotiff_file_out, "w", **out_profile) as dst:
            dst.write(pred_chm_array, 1)

        print("FINISHED!, exported to", geotiff_file_out)
