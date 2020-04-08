#  Train
##  example for 3D model
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

## example for 2D model (since most of them are the same, only the differences are listed here)
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

# Val & Test
When testing video models, we often use intensive sampling and then average logits as the final result.
```yaml
  evaluate:
    spatial_crops: 3 # The number of crops in the spatial dimension
    temporal_samples: 10 # The number of crops in the temporal dimension

# Finally, the number of samples used for testing in each video is 3 * 10 = 30
```
