## install
```
cd mmsegmentation
pip install -e .
```

## download pre-trained 
```
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth pretrain/swin_large_patch4_window12_384_22k.pth
```

## custom configs
[custom configs](https://github.com/boostcampaitech3/level2-semantic-segmentation-level2-cv-18/tree/kihoon/mmsegmentation/configs/custom)


## train
```
python tools/train.py /opt/ml/level2-semantic-segmentation-level2-cv-18/mmsegmentation/configs/swin/swin_l.py
```
