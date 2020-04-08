# X-Temporal

X-Temporal is an open source video understanding codebase from Sensetime X-Lab group that provides state-of-the-art video classification models, including papers "[Temporal Segment Networks](https://arxiv.org/abs/1608.00859)", "[Temporal Interlacing Network](https://arxiv.org/abs/2001.06499)", "[Temporal Shift Module](https://arxiv.org/abs/1811.08383)", "[ResNet 3D](https://arxiv.org/pdf/1711.11248)", "[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)", and "[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)". 

## Introduction
* Support popular video understanding frameworks
  * SLowFast
  * R(2+1)D
  * R3D
  * TSN
  * TIN
  * TSM
* Support various datasets (Kinetics, Something2Something, Multi-Moments in Time...)
  * Take raw video  as input
  * Take video RGB frames as input
  * Take video Flow frames as input
  * Support Multi-label dataset
* High-performance and modular design can help rapid implementation and evaluation of novel video research ideas.
* With the codebase we won the 1st place in the ICCV19 - Multi Moments in Time challenge. [Challenge Website](http://moments.csail.mit.edu/results2019.html).



## Updates
v0.1.0 (08/04/2020)
> X-Temporal is online!

## Get Started
### Prerequisites

The code is built with following libraries:

- [PyTorch](https://pytorch.org/) 1.0 or higher
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [tqdm](https://github.com/tqdm/tqdm.git)
- [sklearn](https://github.com/scikit-learn/scikit-learn)
- [scikit-learn](https://scikit-learn.org/stable/)
- [decord](https://github.com/dmlc/decord)

For extracting frames from video data, you may need [ffmpeg](https://www.ffmpeg.org/).

### Installation
1. clone repo
```bash
git clone https://github.com/Sense-X/X-Temporal.git  X-Temporal
cd X-Temporal
```
2. run the install script
```bash
 ./easy_setup.sh
```


### Prepare dataset
Each row in the meta file of the data set represents a video, which is divided into 3 columns, which are the picture folder, frame number, and category id after the frame extraction. For example as shown below:
```
abseiling/Tdd9inAW1VY_000361_000371 300 0
zumba/x0KPHFRbzDo_000087_000097 300 599
```

You can also directly read the original video file. Decor library is used in X-Temporal code for real-time video frame extraction.
```
abseiling/Tdd9inAW1VY_000361_000371.mkv 300 0
zumba/x0KPHFRbzDo_000087_000097.mkv 300 599
```
In the ** tools ** folder, scripts for extracting frames and generating data set meta files are provided.

### About multi-label classification
The format of the multi-category data set is as follows, which are the video path, the number of frames, and the categories included.
```
trimming/getty-cutting-meat-cleaver-video-id163936215_13.mp4 90 144,246
exercising/meta-935267_68.mp4 92 69
cooking/yt-SSLy25MQb9g_307.mp4 91 264,311,7,188,246
```

YAML config:
```
trainer:
    loss_type: bce
dataset:
    multi_class: True
```

### Training
1. Create a folder for the experiment.
```bash
cd /path/to/X-Temporal
mkdir -p experiments/test
```

2. New or copy config from existing experiment config.
```bash
cp experiments/r2plus1d/default.config experiments/test
cp experiments/r2plus1d/run.sh experiments/test
```

3. Set up training scripts, where ROOT and cfg fiile may need to be changed according to specific settings
```bash
T=`date +%m%d%H%M`
ROOT=../..
cfg=default.yaml

export PYTHONPATH=$ROOT:$PYTHONPATH

python $ROOT/x_temporal/train.py --config $cfg | tee log.train.$T
```

4. Start training.
```bash
./test.sh
```

### Testing
1. Set the resume_model path in config.
```yaml
saver: # Required.
    resume_model: checkpoints/ckpt_e13.pth # checkpoint to test
```
2. Set the parameters in the evaluate in config, such as the need to use multiple crops on the spatial and temporal during the test to modify the specific parameters. (it is recommended to reduce the batchsize by the same proportion)
```yaml
  evaluate:
    spatial_crops: 3
    temporal_samples: 10
```
3. Modify run.sh or create new test.sh, the main modification is to change train.py to test.py. The sample is as follows:
```bash
T=`date +%m%d%H%M`
ROOT=../..
cfg=default.yaml

export PYTHONPATH=$ROOT:$PYTHONPATH

python $ROOT/x_temporal/test.py --config $cfg | tee log.test.$T
```
4. Start Testing
```bash
./test.sh
```

## LICENSE
X-Temporal is released under the [MIT license](LICENSE). 

## Configuration details
###  Train
####  example for 3D model
```yaml
version: 1.0 # version
config:
  dataset:
    workers: 3 # number of workers per process
    num_class: 102 # Total number of dataset categories
    num_segments: 16 # input frames
    batch_size: 32
    img_prefix: 'image _ {: 05d} .jpg' # If you read RGB frames as input, you need to set this parameter to define its naming mode
    video_source: False # Whether to directly read the video 
    dense_sample: True # Whether the data sampling is dense sampling (or uniform sampling)
    modality: RGB # RGB, FLOW
    flow_prefix: ''
    root_dir: / path # The root directory where the dataset files is located
    flip: True # Use flip as augmentation
    dense_sample_rate: 2 # dense sampling rate (sample every n frames)
    input_mean: [0.485, 0.456, 0.406] # comes from imagenet params
    input_std: [0.229, 0.224, 0.225]
    crop_size: 112
    scale_size: 128 # The size after resize the short side of the frame when augmentation
    train:
      meta_file: / path
    val:
      meta_file: / path
    test:
      meta_file: / path

  net:
    arch: stresnet18 # model name and depth
    model_type: 3D # 2D or 3D
    dropout: 0.0
    max_pooling: True # Use maxpooling layer after Conv1 to reduce the spatial size to 1/2 (only work at R(2+1)D models)

  trainer:
    print_freq: 20 # output log every n iter
    eval_freq: 5 # eval every n epochs and output log
    epochs: 120 # total training epochs
    start_epoch: 0
    loss_type: nll
    no_partial_bn: False # FreezeBN (currently only for 2D models)
    clip_gradient: 20 # Gradient crop
    lr_scheduler: # Configuration can refer to pytorch
      warmup_epochs: 10
      type: CosineAnnealingLR
      kwargs:
        T_max: 120
    optimizer: # Configuration can refer to pytorch
      type: SGD
      kwargs:
        lr: 0.4
        momentum: 0.9
        weight_decay: 0.0005
        nesterov: True


  saver:
    save_dir: 'checkpoint /' # checkpoint save path
    pretrain_model: '/ path' # Read pretrain model path
    resume_model: '/ path' # resume model path

```

#### example for 2D model (since most of them are the same, only the differences are listed here)
```yaml
  net:
    arch: resnet50
    model_type: 2D
    shift: True # TSM model switch
    shift_div: 8
    tin: False # TIN model switch
    consensus_type: avg # The consensus function used when calculating each frame needs to be summarized, generally use avg
    dropout: 0.8
    non_local: False
    pretrain: True # imagenet pretrain for 2D network

```

### Val & Test
When testing video models, we often use intensive sampling and then average logits as the final result.
```yaml
  evaluate:
    spatial_crops: 3 # The number of crops in the spatial dimension
    temporal_samples: 10 # The number of crops in the temporal dimension

# Finally, the number of samples used for testing in each video is 3 * 10 = 30
```
