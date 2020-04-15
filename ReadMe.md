# X-Temporal

**Easily implement SOTA video understanding methods with PyTorch on multiple machines and GPUs**

X-Temporal is an open source video understanding codebase from Sensetime X-Lab group that provides state-of-the-art video classification models, including papers "[Temporal Segment Networks](https://arxiv.org/abs/1608.00859)", "[Temporal Interlacing Network](https://arxiv.org/abs/2001.06499)", "[Temporal Shift Module](https://arxiv.org/abs/1811.08383)", "[ResNet 3D](https://arxiv.org/pdf/1711.11248)", "[SlowFast Networks for Video Recognition](https://arxiv.org/abs/1812.03982)", and "[Non-local Neural Networks](https://arxiv.org/abs/1711.07971)". 

*This repo includes all models and codes used in our 1st place solution in ICCV19-Multi Moments in Time Challenge [Challenge Website](http://moments.csail.mit.edu/results2019.html)*

## Introduction
* Support popular video understanding frameworks
  * SlowFast
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



## Updates
v0.1.0 (08/04/2020)
> X-Temporal is online!

## Get started
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
In the **tools** folder, scripts for extracting frames and generating data set meta files are provided.

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

## [ModelZoo](ModelZoo.md)


## LICENSE
X-Temporal is released under the [MIT license](LICENSE). 

## [Configuration details](Configuration.md)

## Reference
Kindly cite our publications if this repo and algorithms help in your research.
```
@article{zhang2020top,
  title={Top-1 Solution of Multi-Moments in Time Challenge 2019},
  author={Zhang, Manyuan and Shao, Hao and Song, Guanglu and Liu, Yu and Yan, Junjie},
  journal={arXiv preprint arXiv:2003.05837},
  year={2020}
}

@article{shao2020temporal,
    title={Temporal Interlacing Network},
    author={Hao Shao and Shengju Qian and Yu Liu},
    year={2020},
    journal={AAAI},
}
```

## Contributors
X-Temporal is maintained by Hao Shao and ManYuan Zhang and [Yu Liu](http://liuyu.us/).
