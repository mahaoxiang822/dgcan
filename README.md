## Installation
The implementation of DGCAN is based on MMDetection.

Please refer to [get_started.md](docs/get_started.md) for installation.

## Dataset
To prepare the dataset,

(1) download [Graspnet-1billion](https://graspnet.net/index.html).

(2) download our refined rectangle label and views from [GoogleDrive](https://drive.google.com/drive/folders/1vavvOjjd3nhs0fiTUpcR_As_dn3OvdCt?usp=sharing).

    -- data
        -- planer_graspnet
            -- scenes
            -- depths
            -- rect_labels_filt_top10%_depth2_nms_0.02_10
            -- views
            -- models
            -- dex_models
    

## Training

1. For training DGCAN, the configuration files are in configs/graspnet/.

```shell script

python tools/train.py configs/graspnet/faster_r2cnn_r50_1016_rgb_ddd_depth_mh_attention_k.py

CUDA_VISIBLE_DEVICES=0,1 .tools/dist_train.sh configs/graspnet/faster_r2cnn_r50_1016_rgb_ddd_depth_mk_attention_k.py 2

```

## Testing

1. For testing DGCAN, only support single-gpu inference.

```shell script
python tools/test_graspnet.py configs/graspnet/faster_r2cnn_r50_1016_rgb_ddd_depth_mk_attention_k.py checkpoints/dgcan_trained.pth --eval grasp
```
Our trained checkpoints can be download from [GoogleDrive](https://drive.google.com/drive/folders/1vavvOjjd3nhs0fiTUpcR_As_dn3OvdCt?usp=sharing).