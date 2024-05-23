# Quickstart

Run in Colab, switch to A100 GPU runtime

Install 

```
git clone https://github.com/facebookresearch/HighResCanopyHeight.git
#git clone https://github.com/chrieke/HighResCanopyHeight.git

cd HighResCanopyHeight
#pip install -r requirements.txt

pip install awscli
pip install rasterio numpy pandas matplotlib
pip install torchvision==0.15.2
pip install pytorch-lightning==1.7
pip install torchmetrics==0.11.4 
pip install torchtext --upgrade
pip install torch --upgrade
```

Download NEON data and pretrained models:
```
aws s3 --no-sign-request cp --recursive s3://dataforgood-fb-data/forests/v1/models/ .
unzip data.zip
```

Run test inference
```
#python inference.py --checkpoint saved_checkpoints/SSLhuge_satellite.pth 
python inference.py --checkpoint saved_checkpoints/SSLhuge_satellite.pth --display True 
#poetry run python inference.py --checkpoint saved_checkpoints/compressed_SSLhuge.pth --display True 
python HighResCanopyHeight/inference2.py --checkpoint ./drive/MyDrive/meta-tree-height/saved_checkpoints/SSLhuge_satellite.pth --display True  
```

Run inference on custom image
https://github.com/facebookresearch/HighResCanopyHeight/issues/3

- test images e.g. 2184x2184, 1766x1766, 1765  something like that.
- 0.59m res
- The 2048px image = 1208m in reality. 
- The model is applied to cutouts of that, 250m, would be around 423px tiles
- Is then applied to tiles of that, 
- RGB only, uint8
- images are not georeferenced
```
#python inference.py --checkpoint saved_checkpoints/SSLhuge_satellite.pth 
poetry run python inference.py --checkpoint saved_checkpoints/SSLhuge_satellite.pth --display True 
#poetry run python inference.py --checkpoint saved_checkpoints/compressed_SSLhuge.pth --display True 
```


# Troubleshooting:
brew install libtiff

#inference.py  72 
ckpt = torch.load(ssl_path, map_location=torch.device('cpu'))   #
If you are running on a CPU-only machine, please use torch.load with map_location=torch.device('cpu') to map your storages to the CPU.

