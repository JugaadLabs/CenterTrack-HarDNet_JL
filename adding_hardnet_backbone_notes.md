# Using a HarDNet backbone in the CenterTrack network

We investigate how to modify the CenterTrack network to use a different backbone better suitable for TensorRT conversion and edge detection. We choose the [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) network for this purpose. To modify the CenterTrack network, we make use of the following observations:
1. CenterTrack is based heavily on the [CenterNet](https://github.com/xingyizhou/CenterNet) network introduced by the same authors, and
2. There already exists a modified [CenterNet-HarDNet](https://github.com/PingoLH/CenterNet-HarDNet) implementation created by the HarDNet authors and was directly forked from the CenterNet repository

We take the following approach to ultimately arrive at a network which we will call CenterTrack-HarDNet, à la CenterNet-HarDNet above:

1. Take advantage of the fact that the CenterNet-HarDNet repository was forked from the CenterNet repository, meaning the directory and file structures are very similar/identical in most places. Utilize VSCode's `Select for Compare` and `Compare with Selected` feature to examine corresponding versions of files in the CenterNet and CenterNet-HarDNet codebases to determined what changes were made to incorporate the HarDNet backbone with CenterNet
2. Try to make analagous changes in the CenterTrack code base which is largely based off of the CenterNet code base and uses very similar structure

We made clean clones of all three repositories (CenterNet, CenterNet-HarDNet, and CenterTrack), a second clone of CenterNet called CenterNet_mod, and a second clone of CenterTrack called CenterTrack_mod.

## Using HarDNet with CenterNet: `CenterNet` $\rightarrow$ `CenterNet-HarDNet` 

If a step calls for modification, right-click the file in the `CenterNet` repo and choose `Select for Compare`, and right-click the corresponding file in the `CenterNet-HarDNet` repo and choose `Compare with Selected` to find a line-by-line comparison of the two files and any changes that were made.

### Adding the backbone and inferencing

Here we try to look for changes that were to the CenterNet repo to use the HarDNet backbone network so that we can run inference on images using the pretrained model available on the `CenterNet-HarDNet` GitHub.

- `CenterNet_mod/src/lib/models/networks/`:
	- Copy `hardnet.py` file from `CenterNet-HarDNet/src/lib/models/networks/`

- `CenterNet_mod/src/`
	- Copied `hardnet68_base.pth` and `hardnet85_base.pth` from `CenterNet-HarDNet/src`

- `CenterNet_mod/src/lib/models/model.py`:
	- add `get_hardnet` function import and add to `_model_factory`

- `CenterNet_mod/src/lib/detectors/ctdet.py`:
	- modify lines in `process` method of `CtdetDetector` class

- `CenterNet_mod/src/lib/detectors/base_detector.py`:
	- modify line in `pre_process` method of `BaseDetector` class

- `CenterNet_mod/src/lib/models/utils.py`:
	- new `gather` function for TensorRT mode
	- change in `_gather_feat` function for TRT flag

- `CenterNet_mod/src/lib/models/decode.py`:
	- Add `trt` kwarg to `_topk` and `ctdet_decode`
	- Add `trt` kwarg value to function calls in those two functions

- `CenterNet_mod/src/lib/opts.py`:
	- Replaced file with `CenterNet-HarDNet/src/lib/opts.py` version (too many changes)

After these changes, was able to inference images using the pre-trained model `centernet_hardnet85_coco_608.pth`. Compared the detection results from the original `CenterNet-HarDNet` repo and the newly modified `CenterNet_mod` repo and found the exact same detections. Compare the two files `centernet-hardnet_result.txt` with `centernet_mod_result.txt`.

### Training the model

Here we look for changes that need to be made to the `CenterNet` repo so that we could use the included training scripts and files to train on our own CenterNet-HarDNet model. (We don't actually need to train a model, we just want to make sure that we understand how the original code base was modified to accomodate training the HarDNet backbone to that we can try to mimic those changes for training CenterTrack with a HarDNet backbone).

To train a model in `CeterNet`, one will need to appropriately set up a dataset. We use COCO 2017, which can be obtained from the [COCO dataset download page](https://cocodataset.org/#download) and consists of separate `.zip` files. The data needs to be either copied or symbolically linked to the `CenterNet/data` directory and organized as 
```
${CenterNet_ROOT}:
|
`-- data
    |
    `-- coco
		|-- annotations
        |   |-- instances_train2017.json
        |   |-- instances_val2017.json
        |   |-- person_keypoints_train2017.json
        |   |-- person_keypoints_val2017.json
        |   `-- image_info_test-dev2017.json
        |-- train2017
        |-- val2017
        `-- test2017
``` 
Since we will be training in multiple repositories, we arrange our downloaded COCO dataset into the same structure as above, and then create symbolic links:
```
ln -s /path/to/downloaded/coco2017 /path/to/CenterNet/data/coco
``` 

Next we test out an example command to start training a model on COCO with the unmodified `CenterNet` repo, to be run inside `CenterNet/src`:
```
python3 main.py ctdet --exp_id coco_dla --batch_size 8 --master_batch 15 --lr 1.25e-4  --gpus 0
```

We needed to reduce the batch size from `32` to `8` because we ran out of GPU memory. This produced the expected terminal output:
```
jugaadlabs@bionic:~/JL/centertrack_testing/CenterNet/src$ python3 main.py ctdet --exp_id coco_dla --batch_size 8 --master_batch 15 --lr 1.25e-4  --gpus 0
Fix size testing.
training chunk_sizes: [15]
The output will be saved to  /home/jugaadlabs/JL/centertrack_testing/CenterNet/src/lib/../../exp/ctdet/coco_dla
heads {'hm': 80, 'wh': 2, 'reg': 2}
Namespace(K=100, aggr_weight=0.0, agnostic_ex=False, arch='dla_34', aug_ddd=0.5, aug_rot=0, batch_size=8, cat_spec_wh=False, center_thresh=0.1, chunk_sizes=[15], data_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet/src/lib/../../data', dataset='coco', debug=0, debug_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet/src/lib/../../exp/ctdet/coco_dla/debug', debugger_theme='white', demo='', dense_hp=False, dense_wh=False, dep_weight=1, dim_weight=1, down_ratio=4, eval_oracle_dep=False, eval_oracle_hm=False, eval_oracle_hmhp=False, eval_oracle_hp_offset=False, eval_oracle_kps=False, eval_oracle_offset=False, eval_oracle_wh=False, exp_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet/src/lib/../../exp/ctdet', exp_id='coco_dla', fix_res=True, flip=0.5, flip_test=False, gpus=[0], gpus_str='0', head_conv=256, heads={'hm': 80, 'wh': 2, 'reg': 2}, hide_data_time=False, hm_hp=True, hm_hp_weight=1, hm_weight=1, hp_weight=1, input_h=512, input_res=512, input_w=512, keep_res=False, kitti_split='3dop', load_model='', lr=0.000125, lr_step=[90, 120], master_batch_size=15, mean=array([[[0.40789655, 0.44719303, 0.47026116]]], dtype=float32), metric='loss', mse_loss=False, nms=False, no_color_aug=False, norm_wh=False, not_cuda_benchmark=False, not_hm_hp=False, not_prefetch_test=False, not_rand_crop=False, not_reg_bbox=False, not_reg_hp_offset=False, not_reg_offset=False, num_classes=80, num_epochs=140, num_iters=-1, num_stacks=1, num_workers=4, off_weight=1, output_h=128, output_res=128, output_w=128, pad=31, peak_thresh=0.2, print_iter=0, rect_mask=False, reg_bbox=True, reg_hp_offset=True, reg_loss='l1', reg_offset=True, resume=False, root_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet/src/lib/../..', rot_weight=1, rotate=0, save_all=False, save_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet/src/lib/../../exp/ctdet/coco_dla', scale=0.4, scores_thresh=0.1, seed=317, shift=0.1, std=array([[[0.2886383 , 0.27408165, 0.27809834]]], dtype=float32), task='ctdet', test=False, test_scales=[1.0], trainval=False, val_intervals=5, vis_thresh=0.3, wh_weight=0.1)
Creating model...
Setting up data...
==> initializing coco 2017 val data.
loading annotations into memory...
Done (t=0.34s)
creating index...
index created!
Loaded val 5000 samples
==> initializing coco 2017 train data.
loading annotations into memory...
Done (t=8.79s)
creating index...
index created!
Loaded train 118287 samples
Starting training...
/home/jugaadlabs/.local/lib/python3.6/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))
ctdet/coco_dla |                                | train: [1][84/14785]|Tot: 0:00:55 |ETA: 2:33:53 |loss 60.3296 |hm_loss 57.9812 |wh_loss 20.8580 |off_loss 0.2626 |Data 0.002s(0.008s) |Net 0.653s
```
On our local machine, it will take over two and a half hours for a single epoch. 

Now we move on and check if any changes were made to the files needed for training in the creation of `CenterNet-HarDNet` from `CenterNet`.

- `CenterNet_mod/src/main.py`:
	- add `derivative_mod` function
    - add lines to `main` after `for epoch in range(start_epoch + 1, opt.num_epochs + 1):`

- `CenterNet_mod/src/lib/trains/base_trainer.py`:
    - add lines to `run_epoch` after `data_time.update(time.time() - end)`

- `CenterNet_mod/src/lib/trains/ctdet.py`:
    - add lines to `run_epoch` after `data_time.update(time.time() - end)`
    - Replaced file with `CenterNet-HarDNet/src/lib/trains/ctdet.py` version (too many changes)

- `CenterNet_mod/src/lib/utils/image.py`:
	- Replaced file with `CenterNet-HarDNet/src/lib/utils/image.py` version (too many changes)

- `CenterNet_mod/src/lib/dataset/sample/ctdet.py`:
	- Replaced file with `CenterNet-HarDNet/src/dataset/sample/ctdet.py` version (too many changes)

After the above modifications in `CenterNet_mod`, running the following command in `CenterNet_mod/src` sucessfully began the training process:
```
python3 main.py ctdet --exp_id coco_h85 --arch hardnet_85 --batch_size 4 --master_batch 24 --lr 1e-2 --gpus 0 --num_workers 16 --num_epochs 300 --lr_step 230,280
```

This produced the output:
```
jugaadlabs@bionic:~/JL/centertrack_testing/CenterNet_mod/src$ python3 main.py ctdet --exp_id coco_h85 --arch hardnet_85 --batch_size 4 --master_batch 24 --lr 1e-2 --gpus 0 --num_workers 16 --num_epochs 300 --lr_step 230,280
Fix size testing.
training chunk_sizes: [24]
The output will be saved to  /home/jugaadlabs/JL/centertrack_testing/CenterNet_mod/src/lib/../../exp/ctdet/coco_h85
heads {'hm': 80, 'wh': 4}
Namespace(K=100, aggr_weight=0.0, agnostic_ex=False, arch='hardnet_85', aug_ddd=0.5, aug_rot=0, batch_size=4, cat_spec_wh=False, center_thresh=0.1, chunk_sizes=[24], data_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet_mod/src/lib/../../data', dataset='coco', debug=0, debug_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet_mod/src/lib/../../exp/ctdet/coco_h85/debug', debugger_theme='white', demo='', dense_hp=False, dense_wh=False, dep_weight=1, dim_weight=1, down_ratio=4, eval_oracle_dep=False, eval_oracle_hm=False, eval_oracle_hmhp=False, eval_oracle_hp_offset=False, eval_oracle_kps=False, eval_oracle_offset=False, eval_oracle_wh=False, exp_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet_mod/src/lib/../../exp/ctdet', exp_id='coco_h85', fix_res=True, flip=0.5, flip_test=False, gpus=[0], gpus_str='0', head_conv=256, heads={'hm': 80, 'wh': 4}, hide_data_time=False, hm_hp=True, hm_hp_weight=1, hm_weight=1, hp_weight=1, input_h=512, input_res=512, input_w=512, keep_res=False, kitti_split='3dop', load_model='', load_trt='', lr=0.01, lr_step=[230, 280], master_batch_size=24, mean=array([[[0.40789655, 0.44719303, 0.47026116]]], dtype=float32), metric='loss', mse_loss=False, nms=False, no_color_aug=False, norm_wh=False, not_cuda_benchmark=False, not_hm_hp=False, not_prefetch_test=False, not_rand_crop=False, not_reg_bbox=False, not_reg_hp_offset=False, not_reg_offset=False, num_classes=80, num_epochs=300, num_iters=-1, num_stacks=1, num_workers=16, off_weight=0.5, output_h=128, output_res=128, output_w=128, pad=31, peak_thresh=0.2, print_iter=0, rect_mask=False, reg_bbox=True, reg_hp_offset=True, reg_loss='l1', reg_offset=True, resume=False, root_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet_mod/src/lib/../..', rot_weight=1, rotate=0, save_all=False, save_dir='/home/jugaadlabs/JL/centertrack_testing/CenterNet_mod/src/lib/../../exp/ctdet/coco_h85', scale=0.4, scores_thresh=0.1, seed=317, shift=0.1, std=array([[[0.2886383 , 0.27408165, 0.27809834]]], dtype=float32), task='ctdet', test=False, test_scales=[1.0], trainval=False, val_intervals=5, vis_thresh=0.3, wh_weight=0.05, wlr=5e-05)
Creating model...
3 x 3 x 3 x 48
3 x 3 x 48 x 96
3 x 3 x 96 x 24
3 x 3 x 120 x 40
3 x 3 x 40 x 24
3 x 3 x 160 x 70
3 x 3 x 70 x 24
3 x 3 x 94 x 40
3 x 3 x 40 x 24
3 x 3 x 230 x 118
Blk out = 214
1 x 1 x 214 x 192
3 x 3 x 192 x 24
3 x 3 x 216 x 40
3 x 3 x 40 x 24
3 x 3 x 256 x 70
3 x 3 x 70 x 24
3 x 3 x 94 x 40
3 x 3 x 40 x 24
3 x 3 x 326 x 118
3 x 3 x 118 x 24
3 x 3 x 142 x 40
3 x 3 x 40 x 24
3 x 3 x 182 x 70
3 x 3 x 70 x 24
3 x 3 x 94 x 40
3 x 3 x 40 x 24
3 x 3 x 444 x 200
Blk out = 392
1 x 1 x 392 x 256
3 x 3 x 256 x 28
3 x 3 x 284 x 48
3 x 3 x 48 x 28
3 x 3 x 332 x 80
3 x 3 x 80 x 28
3 x 3 x 108 x 48
3 x 3 x 48 x 28
3 x 3 x 412 x 138
3 x 3 x 138 x 28
3 x 3 x 166 x 48
3 x 3 x 48 x 28
3 x 3 x 214 x 80
3 x 3 x 80 x 28
3 x 3 x 108 x 48
3 x 3 x 48 x 28
3 x 3 x 550 x 234
Blk out = 458
1 x 1 x 458 x 320
3 x 3 x 320 x 36
3 x 3 x 356 x 62
3 x 3 x 62 x 36
3 x 3 x 418 x 104
3 x 3 x 104 x 36
3 x 3 x 140 x 62
3 x 3 x 62 x 36
3 x 3 x 522 x 176
3 x 3 x 176 x 36
3 x 3 x 212 x 62
3 x 3 x 62 x 36
3 x 3 x 274 x 104
3 x 3 x 104 x 36
3 x 3 x 140 x 62
3 x 3 x 62 x 36
3 x 3 x 698 x 300
Blk out = 588
1 x 1 x 588 x 480
3 x 3 x 480 x 48
3 x 3 x 528 x 82
3 x 3 x 82 x 48
3 x 3 x 610 x 138
3 x 3 x 138 x 48
3 x 3 x 186 x 82
3 x 3 x 82 x 48
3 x 3 x 748 x 236
3 x 3 x 236 x 48
3 x 3 x 284 x 82
3 x 3 x 82 x 48
3 x 3 x 366 x 138
3 x 3 x 138 x 48
3 x 3 x 186 x 82
3 x 3 x 82 x 48
3 x 3 x 984 x 400
Blk out = 784
1 x 1 x 784 x 256
3 x 3 x 768 x 80
3 x 3 x 848 x 136
3 x 3 x 136 x 80
3 x 3 x 984 x 232
3 x 3 x 232 x 80
3 x 3 x 312 x 136
3 x 3 x 136 x 80
3 x 3 x 1216 x 394
Blk out = 714
HarDNet85 Base Model loaded.
1 x 1 x 1498 x 256
3 x 3 x 672 x 64
3 x 3 x 736 x 108
3 x 3 x 108 x 64
3 x 3 x 844 x 184
3 x 3 x 184 x 64
3 x 3 x 248 x 108
3 x 3 x 108 x 64
3 x 3 x 1028 x 314
Blk out = 570
1 x 1 x 1028 x 192
3 x 3 x 480 x 48
3 x 3 x 528 x 82
3 x 3 x 82 x 48
3 x 3 x 610 x 138
3 x 3 x 138 x 48
3 x 3 x 186 x 82
3 x 3 x 82 x 48
3 x 3 x 748 x 236
Blk out = 428
1 x 1 x 642 x 96
3 x 3 x 288 x 28
3 x 3 x 316 x 48
3 x 3 x 48 x 28
3 x 3 x 364 x 80
Blk out = 136
Parameters= 37274988
Setting up data...
==> initializing coco 2017 val data.
loading annotations into memory...
Done (t=0.34s)
creating index...
index created!
Loaded val 5000 samples
==> initializing coco 2017 train data.
loading annotations into memory...
Done (t=8.74s)
creating index...
index created!
Loaded train 118287 samples
Starting training...
ctdet/coco_h85 |                                | train: [1][1/29571]|Tot: 0:00:08 |ETA: 2 days, 19:07:27 |loss 8664.5405 |hm_loss 8661.5649 |wh_loss 49.6460 |off_loss 0.9865 |Data 0.001s(0.787s) |Net 4.2ctdet/coco_h85 |                                | train: [1][2/29571]|Tot: 0:00:08 |ETA: 1 day, 10:46:56 |loss 8474.9943 |hm_loss 8472.6156 |wh_loss 38.3302 |off_loss 0.9244 |Data 0.004s(0.526s) |Net 2.92ctdet/coco_h85 |                                | train: [1][3/29571]|Tot: 0:00:09 |ETA: 1 day, 0:00:13 |loss 6423.9527 |hm_loss 6421.1753 |wh_loss 45.2994 |off_loss 1.0248 |Data 0.004s(0.396s) |Net 2.265ctdet/coco_h85 |                                | train: [1][4/29571]|Tot: 0:00:09 |ETA: 18:36:44 |loss 5141.0279 |hm_loss 5138.5838 |wh_loss 39.3486 |off_loss 0.9532 |Data 0.004s(0.317s) |Net 1.872s     ctdet/coco_h85 |                                | train: [1][5/29571]|Tot: 0:00:09 |ETA: 15:22:43 |loss 4285.8611 |hm_loss 4283.6291 |wh_loss 35.3568 |off_loss 0.9282 |Data 0.005s(0.265s) |Net 1.609s     ctdet/coco_h85 |                                | train: [1][6/29571]|Tot: 0:00:09 |ETA: 13:13:13 |loss 3674.9717 |hm_loss 3672.9066 |wh_loss 32.2936 |off_loss 0.9008 |Data 0.005s(0.228s) |Net 1.421s     ctdet/coco_h85 |                                | train: [1][7/29571]|Tot: 0:00:10 |ETA: 11:40:43 |loss 3216.8437 |hm_loss 3214.9129 |wh_loss 30.0608 |off_loss 0.8556 |Data 0.005s(0.200s) |Net 1.281s     ctdet/coco_h85 |                                | train: [1][8/29571]|Tot: 0:00:10 |ETA: 10:31:21 |loss 2860.7416 |hm_loss 2858.7240 |wh_loss 31.7398 |off_loss 0.8611 |Data 0.004s(0.178s) |Net 1.172s  
```

# Using HarDNet with CenterTrack: `CenterTrack` $\rightarrow$ `CenterTrack-HarDNet`

Now that we have understood how to incorporate the HarDNet backbone network with the CenterNet detection head, we apply what we learned, along with the observations listed above, to try to make similar modifications in the `CenterTrack` code base to include the HarDNet backbone. We start with two freshly cloned versions of the `CenterTrack` repo, one to keep unmodified to use to compare files, and one renamed `CenterTrack_mod` in which we make changes.

## Code Execution in the current model (DLA backbone)

First we need to understand how a single image works its way through the various pieces of the code base. This is important becuase it will tell us which parts we need to take special notice of when we incorporate a new backbone. To do this, we create a new file, named `centertrack_mod_sandbox.py` which we make and run inside the `CenterTrack_mod/src` directory. In the following, outside of code blocks, we try to replace `self.some_method` with `ClassName.some_method` to increase clarity to which class we are refering to. 

First off, to inference images, we need the following imports:
```python
from detector import Detector
from opts import opts
```
These import the `Detector` class from `CenterTrack_mod/src/lib/detector.py` and `CenterTrack_mod/src/lib/opts.py`. The `Detector` class is initialized with the `opts` instance containing cutomized options for the specific use case. For running the DLA network, we download the pretrained model `coco_tracking.pth` and use the `--load_model` flag to point to its location. We also set `--input_w=704`, and `--input_h=608` for our specific image size of `720x616`. The values needs to be divisible by 32, because in the DLA model as we will see, the input resolution get's downsampled by 2 at each level of a five level backbone. We choose the greatest multiple of 32 that is equal or less than our actual image resolution. A printout of some of the options to verify things were set correctly is as follows:
```
===== opt properties =====
opt.arch: dla_34
opt.task: task=tracking
opt.heads: {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}
opt.head_conv: {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
opt.input_h: 608
opt.input_w: 704
==========================
```

Next, we use the `run` method of the `Detector` class to initiate the inference on the image, returning a dictionary of results and timing info. For the test image we choose, the `'result'` value of the returned dictionary is:
```
{'score': 0.6022488, 'class': 3, 'ct': array([171.81818, 132.09091], dtype=float32), 'tracking': array([ 1.8437347, -2.1887512], dtype=float32), 'bbox': array([149.76167, 119.34333, 196.64006, 146.76524], dtype=float32), 'tracking_id': 1, 'age': 1, 'active': 1}
```
Now let us take a deep look into what happens to the `image` passed inside the `Detector.run(image)` call.

1. Firstly, when the detector is initialized, inside it's `__init__` method, we create and load the model specified by the initialized `opt` instance's `--arch` flag. The default is `dla_34`, corresponding to our pre-trained model we have been using. These steps are done by the imported `create_model` and `load_model` functions from `CenterTrack_mod/src/lib/model/model.py`. Inside the `Detector.__init__()` we have:
    ```python
    self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    self.model = load_model(self.model, opt.load_model, opt)
    ```
    where `create_model` is defined as 
    ```python
    def create_model(arch, head, head_conv, opt=None):
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        arch = arch[:arch.find('_')] if '_' in arch else arch
        model_class = _network_factory[arch]
        model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
        return model
    ```
    The `model_class` is defined via a dictionary keyed by the network architecture:
    ```python
    _network_factory = {
        'resdcn': PoseResDCN,
        'dla': DLASeg,
        'res': PoseResNet,
        'dlav0': DLASegv0,
        'generic': GenericNetwork
    }
    ```
    The result of this, for `--arch=dla_34` is `self.model` being an instance of `DLASeg`, so that later when an image is passed, e.g., as `self.model(...)`, the input is implicitely passed to the `forward` method of the `DLASeg` class. We will look at this more in depth later.

2. The `run` method's function signature is 
    ```python
    def run(self, image_or_path_or_tensor, meta={}):
    ```
    The positional argument can take an image as a `numpy` array (loaded with OpenCV's `imread`) or a path to the image file (in which case the image is read in with OpenCV), or a pytorch tensor containing the image.

3. Next, the image is passed to the `Detector.pre_process()` method where different transformations are applied depending on the options set in the `opts` instance the `Detector` was initialized with. We insert print statments at the beginning and end of the `pre_process` to print out the shape of an image passed to it and that returned from it:
```
in detector.py preprocess: input image.shape: (616, 720, 3)
in detector.py preprocess: images[0].shape: torch.Size([3, 608, 704])
```
As expected, our input image is resized to the dimensions specified by `--input_h(w)`. The ordering of the dimensions is typical for a pytorch tensor representing an image: `[channels, height, width]`.

4. After pre-processing comes the initialization of a `Tracker` class, imported from `CenterTrack_mod/src/lib/utils/tracker.py`. Since we are running a single image, the previous image is set to the current image. The previous heatmap is also generated from the previous image if the option is set.

5. Finally, the current (and prior, if set) image is run through the model via the `Detector.process()` method. The forward pass is contained within the `pytorch` context `with torch.no_grad()`:    
    ```python
    def process(self, images, pre_images=None, pre_hms=None,
        pre_inds=None, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images, pre_images, pre_hms)[-1]
    ```
    The last line above with `self.model(...)` implicitely calls the forward method of `DLASeg`, as explained in step 1 above. Now this is when things get interesting. First let us look at how `DLASeg` is defined, inside `CenterTrack_mod/src/lib/model/networks/dla.py`:
    ```python
    class DLASeg(BaseModel):
        def __init__(self, num_layers, heads, head_convs, opt):
            super(DLASeg, self).__init__(heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
            down_ratio=4
            self.opt = opt
            self.node_type = DLA_NODE[opt.dla_node]
            print('Using node type:', self.node_type)
            self.first_level = int(np.log2(down_ratio))
            self.last_level = 5
            self.base = globals()['dla{}'.format(num_layers)](pretrained=(opt.load_model == ''), opt=opt)

            channels = self.base.channels
            scales = [2 ** i for i in range(len(channels[self.first_level:]))]
            self.dla_up = DLAUp(
                self.first_level, channels[self.first_level:], scales,
                node_type=self.node_type)
            out_channel = channels[self.first_level]

            self.ida_up = IDAUp(
                out_channel, channels[self.first_level:self.last_level], 
                [2 ** i for i in range(self.last_level - self.first_level)],
                node_type=self.node_type)  
    ```
    The only other methods defined are `img2feats` and `imgpre2feats`. The `DLASeg` class inherits its `forward` method from the class inheritance of `BaseModel`, defined in `CenterTrack_mod/src/lib/model/networks/base_model.py`. This class defines the head of the network and a generic `forward` method that calls either `img2feats` or `imgpre2feats` (which need to be overloaded by the class inheriting from `BaseModel`).
    ```python
    class BaseModel(nn.Module):
        def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
            super(BaseModel, self).__init__()
            if opt is not None and opt.head_kernel != 3:
                print('Using head kernel:', opt.head_kernel)
                head_kernel = opt.head_kernel
            else:
                head_kernel = 3
            self.num_stacks = num_stacks
            self.heads = heads
            for head in self.heads:
                classes = self.heads[head]
                head_conv = head_convs[head]
                if len(head_conv) > 0:
                    out = nn.Conv2d(head_conv[-1], classes, 
                            kernel_size=1, stride=1, padding=0, bias=True)
                    conv = nn.Conv2d(last_channel, head_conv[0],
                                    kernel_size=head_kernel, 
                                    padding=head_kernel // 2, bias=True)
                    convs = [conv]
                    for k in range(1, len(head_conv)):
                        convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                                    kernel_size=1, bias=True))
                    if len(convs) == 1:
                        fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
                    elif len(convs) == 2:
                        fc = nn.Sequential(
                        convs[0], nn.ReLU(inplace=True), 
                        convs[1], nn.ReLU(inplace=True), out)
                    elif len(convs) == 3:
                        fc = nn.Sequential(
                            convs[0], nn.ReLU(inplace=True), 
                            convs[1], nn.ReLU(inplace=True), 
                            convs[2], nn.ReLU(inplace=True), out)
                    elif len(convs) == 4:
                        fc = nn.Sequential(
                            convs[0], nn.ReLU(inplace=True), 
                            convs[1], nn.ReLU(inplace=True), 
                            convs[2], nn.ReLU(inplace=True), 
                            convs[3], nn.ReLU(inplace=True), out)
                    if 'hm' in head:
                        fc[-1].bias.data.fill_(opt.prior_bias)
                    else:
                        fill_fc_weights(fc)
                else:
                    fc = nn.Conv2d(last_channel, classes, 
                        kernel_size=1, stride=1, padding=0, bias=True)
                    if 'hm' in head:
                        fc.bias.data.fill_(opt.prior_bias)
                    else:
                        fill_fc_weights(fc)
                self.__setattr__(head, fc)

        def img2feats(self, x):
            raise NotImplementedError
        
        def imgpre2feats(self, x, pre_img=None, pre_hm=None):
            raise NotImplementedError

        def forward(self, x, pre_img=None, pre_hm=None):
            if (pre_hm is not None) or (pre_img is not None):
                feats = self.imgpre2feats(x, pre_img, pre_hm)
            else:
                feats = self.img2feats(x)
            out = []
            if self.opt.model_output_list:
                for s in range(self.num_stacks):
                    z = []
                    for head in sorted(self.heads):
                        z.append(self.__getattr__(head)(feats[s]))
                    out.append(z)
            else:
                for s in range(self.num_stacks):
                    z = {}
                    for head in self.heads:
                        z[head] = self.__getattr__(head)(feats[s])
                    out.append(z)
            return out
    ```
    The values of the initialization variables for `BaseModel` are defined in the call inside `DLASeg.__init__()`:
    ```python
    super(DLASeg, self).__init__(heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
    ```
    The values of `heads` and `head_convs` come from those passed through `create_model` (see Step 1 above):
    ```python
    model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
    ```
    where _these_ values of `head` and `head_conv` come from the values initialized in `opts` and passed in `Detector.__init__()`:
    ```python
    self.model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
    ```
    A print out of these yields
    ```
    opt.heads: {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}
    opt.head_conv: {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
    ```
    Finally, we can examine the actual head layers by putting print statements at the end of the `for head in self.heads:` loop in `BaseModel.__init__`. This gives a printout:
    ```
    BaseModel for loop head: hm
    self.__getattr__(head): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(256, 80, kernel_size=(1, 1), stride=(1, 1))
    )
    BaseModel for loop head: reg
    self.__getattr__(head): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    )
    BaseModel for loop head: wh
    self.__getattr__(head): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    )
    BaseModel for loop head: tracking
    self.__getattr__(head): Sequential(
    (0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace=True)
    (2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    )
    ```
    Each head layer has in initial 64x3x3 convolution with 256 filters, followed by a ReLU activation, and then a final output convolution of with a number of filters determined by the number of outputs being regressed. For `hm` (keypoint heatmaps) there are 80 outputs (corresponding to the 80 COCO classes), for `reg` (keypoint local offsets) there are 2 for a two dimensional vector, for `wh` (bounding box size) there are 2 for the two box dimensions, and for `tracking` (offset between centers of corresponding objects in current and prior frames) there are 2 for a two dimensional vector.

    Let us continue with our dive into `Detector.run()`.

6. Back in `Detector.run()`'s call to `Detector.process()`, we had the call `self.model(images, pre_images, pre_hms)`. This gets translated to `DLASeg(images, pre_images, pre_hms)` which gets its `forward` method from `BaseModel` to become `BaseModel.forward(images, pre_images, pre_hms)`. Looking at the code above, this call get's initially routed via the first `if` statement to `feats = self.imgpre2feats(x, pre_img, pre_hm)`. This function was defined back in `DLASeg`:  
    ```python
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
        x = self.base(x, pre_img, pre_hm)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i].clone())
        self.ida_up(y, 0, len(y))

        return [y[-1]]
    ```
    Notice how the `pre_img` and `pre_hm` are passed to `self.base` only. This means that the combination of the prior image and/or heatmap is done in `DLASeg.base`. Looking back at how `DLASeg` is initialized, we see the definition of `self.base` as
    ```python
    self.base = globals()['dla{}'.format(num_layers)](pretrained=(opt.load_model == ''), opt=opt)
    ```
    This structure allows different types of DLA networks to serve as the base network, depending on the value of `num_layers`. For our case, `num_layers=34`, so we have our `DLASeg.base` equal to the return value of 
    ```python
    def dla34(pretrained=True, **kwargs):  # DLA-34
        model = DLA([1, 1, 1, 2, 2, 1],
                    [16, 32, 64, 128, 256, 512],
                    block=BasicBlock, **kwargs)
        if pretrained:
            model.load_pretrained_model(
                data='imagenet', name='dla34', hash='ba72cf86')
        else:
            print('Warning: No ImageNet pretrain!!')
        return model
    ```
    Note the `pretrained=(opt.load_model == '')` in the definition of `DLASeg.base` actually results in `pretrained=False` since we pass a file path to `--load_model`.

    So now we know that inside `DLASeg.imgpre2feats`, we are actually calling `DLA(x, pre_img, pre_hm)` where this `DLA` is initialized as in the `dla34` function just above. Let us examine this class's definition:
    ```python
    class DLA(nn.Module):
        def __init__(self, levels, channels, num_classes=1000,
                    block=BasicBlock, residual_root=False, linear_root=False,
                    opt=None):
            super(DLA, self).__init__()
            self.channels = channels
            self.num_classes = num_classes
            self.base_layer = nn.Sequential(
                                nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True))
            self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
            self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
            self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
            self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
            self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
            self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True, root_residual=residual_root)
            if opt.pre_img:
                self.pre_img_layer = nn.Sequential(
                                        nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                        nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True))
            if opt.pre_hm:
                self.pre_hm_layer = nn.Sequential(
                                        nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                        nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                                        nn.ReLU(inplace=True))
    ```
    We have a `DLA.base_layer`, `DLA.pre_img_layer`, and `DLA.pre_hm_layer` which are all defined exactly the same way, consisting of convolutional layers that take the input images with 3 channels and produce a feature map 16 channels deep (since `channels=[16, 32, 64, 128, 256, 512]` from the initializaiton in `dla34`). They are used in the `DLA.forward` method which is the endpoint of the `DLASeg.base()` call inside `DLASeg.imgpre2feats`. In the forward method, the current image is run through `DLA.base_layer`, and the previous image is run through `DLA.pre_img_layer`, with the results added together and then passed to the rest of the network:
    ```python
    def forward(self, x, pre_img=None, pre_hm=None):
        y = []
        x = self.base_layer(x)
        if pre_img is not None:
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            x = x + self.pre_hm_layer(pre_hm)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)       
        return y
    ```
    Since we have finally reached the real crux of how CenterTrack is processing the previous image information, let us insert some print statements to study the shape of the tensors as they are routed through this method:
    ```python
    def forward(self, x, pre_img=None, pre_hm=None):
        print(f'\n============== In dla.py DLA.forward: ================\n')
        print(f'Input x.shape: {x.shape}')
        y = []    
        x = self.base_layer(x)
        print(f'DLA self.base_layer(x).shape: {x.shape}\n')
        if pre_img is not None:
            print(f'pre_img.shape: {pre_img.shape}')
            print(f'DLA self.pre_img_layer(pre_img).shape: {self.pre_img_layer(pre_img).shape}\n')
            x = x + self.pre_img_layer(pre_img)
        if pre_hm is not None:
            print(f'pre_hm.shape: {pre_hm.shape}')
            print(f'DLA self.pre_hm_layer(pre_hm).shape: {self.pre_hm_layer(pre_hm).shape}\n')
            x = x + self.pre_hm_layer(pre_hm)
        print('Entering DLA.forward for loop after combination with pre_img_layer/pre_hm_layer:')
        print(f'Before loop: x.shape: {x.shape}')
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            print(f'self.level{i}(x).shape: {x.shape}')
            y.append(x)
        print(f'\n============== End of DLA forward pass ===============\n')
        return y
    ```
    This produces the following output:
    ```
    ============== In dla.py DLA.forward: ================

    Input x.shape: torch.Size([1, 3, 608, 704])
    DLA self.base_layer(x).shape: torch.Size([1, 16, 608, 704])

    pre_img.shape: torch.Size([1, 3, 608, 704])
    DLA self.pre_img_layer(pre_img).shape: torch.Size([1, 16, 608, 704])

    Entering DLA.forward for loop after combination with pre_img_layer/pre_hm_layer:
    Before loop: x.shape: torch.Size([1, 16, 608, 704])
    self.level0(x).shape: torch.Size([1, 16, 608, 704])
    self.level1(x).shape: torch.Size([1, 32, 304, 352])
    self.level2(x).shape: torch.Size([1, 64, 152, 176])
    self.level3(x).shape: torch.Size([1, 128, 76, 88])
    self.level4(x).shape: torch.Size([1, 256, 38, 44])
    self.level5(x).shape: torch.Size([1, 512, 19, 22])

    ============== End of DLA forward pass ===============
    ```
    You can see the input `x` and `pre_img` are the same shape before and after they are run through `DLA.base_layer` and `DLA.pre_img_layer` respectively, and thus can be added together before passing though the rest of the `DLA` network. At each successive layer in the `DLA` network, the output shape of the tensor gets deeper in channels while becoming smaller in width and height, which is typical behavior of a deep CNN for image analysis. The exception is the first level, `DLA.level0`, which is in fact designed to have the same number of output channels as input channels, 16 in this case, as shown above in the definition of the `DLA` class. Here we see that we need the input resolution set by `--input_h(w)` to be divisible by 32. 

    The output of this `DLA.forward` method is a list of tensors whose shapes are printed out in the above terminal output. Function control returns to `DLASeg.imgpre2feats` which then pass the list of tensors to `DLASeg.dla_up`. This is another part of the DLA network which returns another list of tensors which get modified in-place in the last line of `DLASeg.imgpre2feats` by `DLASeg.ida_up`. The last tensor in the modified list is returned from `DLASeg.imgpre2feats` as `feats` in `BaseModel.forward`. Inserting some print functions, we can examine it:
    ```
    In base_model.py BaseModel.forward after imgpre2feats or img2feats:
    len(feats): 1
    feats[0].shape: torch.Size([1, 64, 152, 176])
    ```
    After this, `feats` is passed through the head of the network: 
    ```python
    for s in range(self.num_stacks):
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(feats[s])
        out.append(z)
    ```
    where for the DLA case, the `for` loop is a single step since `num_stacks = 1` in the initialization of `BaseModel` (See Step 5 above). The actual convolutional layers for each head are examined in step 5 above, and we can verify the shape of the output by printing out `z[head].shape` before the `out.append(z)` line:
    ```
    head: hm
    z[head].shape: torch.Size([1, 80, 152, 176])
    head: reg
    z[head].shape: torch.Size([1, 2, 152, 176])
    head: wh
    z[head].shape: torch.Size([1, 2, 152, 176])
    head: tracking
    z[head].shape: torch.Size([1, 2, 152, 176])
    ```
    The outputs are `152x176` by a number of channels corresponding to the number of outputs needed for each head target. The numbers 152 and 176 comes from the input size of 608 and 704 divided by a downsampling factor of 4, as described in the CenterTrack paper in Section 3 Preliminaries. the tensors whose shapes are printed above are the final output of the `Detector.model(images, pre_images, pre_hms)` inside the `Detector.process()` method. The heatmaps `hm` output is processed with a sigmoid, and then the outputs are send to a decoding function `generic_decode` imported from `CenterTrack_mod/src/lib/model/decode.py` which produced the detections (including confidence score, COCO class id, object center and bounding box, a tracking vector and tracking id unique to this object across frames, and two integers representing the age of the detection and active state). The rest of the `Detector.run()` method involves gathering the results together and producing any visualizations specified in the `opts` instance. The final result looks like
    ```
    {'score': 0.6022488, 'class': 3, 'ct': array([171.81818, 132.09091], dtype=float32), 'tracking': array([ 1.8437347, -2.1887512], dtype=float32), 'bbox': array([149.76167, 119.34333, 196.64006, 146.76524], dtype=float32), 'tracking_id': 1, 'age': 1, 'active': 1}
    ```

## Adding the HarDNet backbone

Let us summarize what we've learned so far by studying the `DLA` model and take note of what types of modifications will be needed to use the HarDNet backbone instead. This backbone is defined in `CenterNet-HarDNet/src/lib/models/networks/hardnet.py` so the first step is to copy that file to `CenterTrack_mod/src/lib/model/networks/`. It contains a clss `HarDNetSeg` which we will modify into the analog of `DLASeg` that we've been studying. Here are the main points we've learned about how CenterTrack inferences with the DLA model: 

1. The different models are accessible to the `Detector` class via the `_network_factory` dictionary inside `CenterTrack_mod/src/lib/model/model.py`, shown in Step 1 in the previous section. This selects the model class (`DLASeg`) to build the network based on the `--arch` flag. 

2. The class `DLASeg` is based off the class `BaseModel` which build's `CenterTrack`'s detection head. An image passed to `DLASeg` gets implicity sent to `BaseModel.forward` which routes it back to `DLASeg.img2feats` or `DLASeg.imgpre2feats`. 

3. In `DLASeg.imgpre2feats`, the current image and previous image are bassed to the `DLASeg.base` class (an instance of `DLA`) and run through separate `DLA.base_layer` and `DLA.pre_img_layer` layers before being added together. Then the combination is run through the rest of the network, generating a `feats` list of tensors which gets sent back to `BaseModel.forward` to be passed through the network's head convolutions.

We first begin by examining the `HarDNetSeg` class in detail as it is originally coded in `hardnet.py`. To make this model accessible to the `Detector` class, we add another key, `'hardnet'` to the `_network_factory` dictionary, with the `HarDNetSeg` class as its value, and import that class from the newly copied `CenterTrack_mod/src/lib/model/networks/hardnet.py`.

### The `HarDNetSeg` class

First, we copy the original file `CenterNet-HarDNet/src/lib/models/networks/hardnet.py` to `CenterTrack_mod/src/lib/model/networks/hardnet.py`. We start by adding the `HarDNetSeg` to the `_network_factory` dictionary and import the class in `CenterTrack_mod/src/lib/model/model.py`:
```python
from .networks.hardnet import HarDNetSeg # NEW line

_network_factory = {
  'hardnet': HarDNetSeg, # NEW line
  'resdcn': PoseResDCN,
  'dla': DLASeg,
  'res': PoseResNet,
  'dlav0': DLASegv0,
  'generic': GenericNetwork
}
```
Now, examining the call signature of `create_model` tells us how to structure the arguments in the `__init__` of `HarDNetSeg`:
```python
def create_model(arch, head, head_conv, opt=None):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model_class = _network_factory[arch]
    model = model_class(num_layers, heads=head, head_convs=head_conv, opt=opt)
    return model
```
Here is the first few lines in definition of `DLASeg`:
```python
class DLASeg(BaseModel):
    def __init__(self, num_layers, heads, head_convs, opt):
        super(DLASeg, self).__init__(heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
```

The orginal `HarDNetSeg` has the analagous first few lines:
```python
class HarDNetSeg(nn.Module):
    def __init__(self, num_layers, heads, pretrained, down_ratio, final_kernel,
                 last_level, head_conv, out_channel=0, trt=False):
        super(HarDNetSeg, self).__init__()
```
In `CenterNet-HarDNet/src/lib/models/model.py`, a wrapper function is used to initialize and return this class to `create_model`:
```python
from .networks.hardnet import get_pose_net as get_hardnet

_model_factory = {
  'hardnet': get_hardnet,
}
```
where `get_pose_net` is defined in `CenterNet-HarDNet/src/lib/models/networks/hardnet.py`:
```python
def get_pose_net(num_layers, heads, head_conv=256, down_ratio=4, trt=False):
    model = HarDNetSeg(
                    num_layers,
                    heads,
                    pretrained=True,
                    down_ratio=down_ratio,
                    final_kernel=1,
                    last_level=4,
                    head_conv=head_conv,
                    trt = trt)
    total_params = sum(p.numel() for p in model.parameters())
    print( "Parameters=", total_params )  
    return model
```
We can use the values defined in this `get_pose_net` to turn some of the positional arguments in `HarDNetSeg` into keyword arguments:
```python
class HarDNetSeg(nn.Module):
	def __init__(self, num_layers, heads, head_convs, opt, 
				 pretrained=True, down_ratio=4, final_kernel=1, last_level=4, out_channel=0, trt=False):
		super(HarDNetSeg, self).__init__()
```

Included in the `HarDNetSeg.__init__()`, we have a line that defines the base of the model:  
```python
self.base = HarDNetBase(num_layers).base
```
This is analagous to how the base of the `DLASeg` model was a specific type of `DLA` class. Here the differentiating factor is the number of layers, which for this implementation of HarDNet, is either `68` or `85`. The rest of the `init` method sets up complicated skip connection structure inherent to the HarDNet architecture. The last part of the `init` function is the definition of the detection head:
```python
self.heads = heads
for head in self.heads:
    classes = self.heads[head]
    if head_conv > 0:
        ch = max(128, classes*4)
        fc = nn.Sequential(
                nn.Conv2d(prev_ch, ch, kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True))
        fill_fc_weights(fc)
        if 'hm' in head:
        fc[-1].bias.data.fill_(-2.19)
    else:
        fc = nn.Conv2d(channels[self.first_level], classes, kernel_size=final_kernel, stride=1, padding=final_kernel // 2, bias=True)
        fill_fc_weights(fc)
        if 'hm' in head:
        fc.bias.data.fill_(-2.19)
    self.__setattr__(head, fc)
```

Recall, this file came from the `CenterNet-HarDNet` repo, so this is CenterNet's detection head. However, we know `self.heads` is defined by that passed to `HarDNetSeg` from `create_model`, which is `opt.heads` and thus contains the detection heads for CenterTrack. We can expect this code block will create layers for each head. Note that if this is true, then we do not need to inherit from `BaseModel` in our implementation. Let's leave the head structure as is for now and keep moving through `HarDNetSeg`.

The last method of `HarDNetSeg` is its `forward` method. Recall, `DLASeg` did not have a `forward` method, and used the one from `BaseModel` which directed input to `DLASeg.img2feats` or `DLASeg.imgpre2feats` which themselves returned the feature maps to `BaseModel.forward` to run them through the detection head. The `HarDNetSeg.forward` method includes the code to run an input image all the way through the network, including its detection head:
```python
def forward(self, x):
    xs = []
    x_sc = []
    
    for i in range(len(self.base)):
        x = self.base[i](x)
        if i in self.skip_nodes:
            xs.append(x)
    
    x = self.last_proj(x)
    x = self.last_pool(x)
    x2 = self.avg9x9(x)
    x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
    x = torch.cat([x,x2,x3],1)
    x = self.last_blk(x)
    
    for i in range(3):
        skip_x = xs[3-i]
        x = self.transUpBlocks[i](x, skip_x, (i<self.skip_lv))
        x = self.conv1x1_up[i](x)
        if self.SC[i] > 0:
            end = x.shape[1]
            x_sc.append( x[:,end-self.SC[i]:,:,:].contiguous() )
            x = x[:,:end-self.SC[i],:,:].contiguous()
        x2 = self.avg9x9(x)
        x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
        x = torch.cat([x,x2,x3],1)
        x = self.denseBlocksUp[i](x)

    scs = [x]
    for i in range(3):
        if self.SC[i] > 0:
        scs.insert(0, F.interpolate(x_sc[i], size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True))

    x = torch.cat(scs,1)
    z = {}
    for head in self.heads:
        z[head] = self.__getattr__(head)(x)
    if self.trt:
        return [z[h] for h in z]
    return [z]
```
The beginning lines run the input through the base of the network:
```python
for i in range(len(self.base)):
    x = self.base[i](x)
    if i in self.skip_nodes:
        xs.append(x)
```
The middle part deals with the skip connections. The final part is the same as the final lines in the `forward` method of `BaseModel`. We can surmise from this that the value of `x` at the line `x = torch.cat(scs,1)` is the analogy of the return value of `DLASeg.img2feats` or `DLASeg.imgpre2feats`.

Let us insert some print statements into the block of code constructing the detection head in `HarDNetSeg.__init__` to print out the head convolution structures:
```
HarDNetSeg for loop head: hm
self.__getattr__(head): Sequential(
  (0): Conv2d(200, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))
)
HarDNetSeg for loop head: reg
self.__getattr__(head): Sequential(
  (0): Conv2d(200, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1))
)
HarDNetSeg for loop head: wh
self.__getattr__(head): Sequential(
  (0): Conv2d(200, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1))
)
HarDNetSeg for loop head: tracking
self.__getattr__(head): Sequential(
  (0): Conv2d(200, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(128, 2, kernel_size=(1, 1), stride=(1, 1))
)
```
Compare this to what we found using `DLASeg` which gets its head layers from `BaseModel`'s `init`:
```
BaseModel for loop head: hm
self.__getattr__(head): Sequential(
(0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(256, 80, kernel_size=(1, 1), stride=(1, 1))
)
BaseModel for loop head: reg
self.__getattr__(head): Sequential(
(0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
)
BaseModel for loop head: wh
self.__getattr__(head): Sequential(
(0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
)
BaseModel for loop head: tracking
self.__getattr__(head): Sequential(
(0): Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
(1): ReLU(inplace=True)
(2): Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
)
```
The only difference is the number of input and output channels. The output channels are controlled by the line with `ch = max(128, classes*4)`. For the heatmap head, `classes=80`, so that layer has `320` output channels. The rest have `classes=2`, resulting in `128` output channels. The input channels are controlled by the variable `prev_ch` which is the output channels of each `HarDBlk` sub-block. The last time it is modified before the head convolutions are defined is in a line `prev_ch += self.SC[0] + self.SC[1] + self.SC[2]`, where the number of skip connections are added to the number of output channels in the final `HarDBlk` sub-block. Its value after this line is `200`. In the case of `DLASeg`, the number of input channels was defined in the initialization of `BaseModel`:
```python
super(DLASeg, self).__init__(heads, head_convs, 1, 64 if num_layers == 34 else 128, opt=opt)
```
For the DLA-34 model, the head convolutions' input channels was `64`. The output channels were controlled by the value of the parameter flag `--head_conv` in the initialization of `opt`. For the `DLA` architecture, this value is `256` which we see in the above printouts.

At this point in our examination of `HarDNetSeg` we can conclude that we should not try to use the same inheritance of `BaseModel` to build the network head and direct the flow of input through its `forward` method. This is because the structure of `HarDNetSeg`'s detection head is different for the different heads, and it doesn't make sense to modify the `BaseModel` class if we already have the head structure in place. Similarly, the `HarDNetSeg`'s forward method provides the passage of input all the way from the base network through the detection heads. It doesn't really make sense to create analagous versions of `DLASeg.img2feats` or `DLASeg.imgpre2feats` unless we later find we need access to the feature maps prior to sending them through the network head.

In this spirit, we only really need to determine how to correctly accomodate the prior image information into the existing `HarDNetSeg.forward` method, so that when we get to the line in `Detector.process` that actually sends the image and prior image to the model, they are handled correctly. In `DLASeg`, the combination of the current and previous images occured at the very start of the forward pass, in the `DLASeg.base`'s `forward` method. For `DLASeg`, `DLASeg.base` was an instance of the `DLA` class, which had special layers specifically for the prior image info. There was a `DLA.base_layer`:
```python
self.base_layer = nn.Sequential(
                    nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                    nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True))
```
and layers for the prior image and heatmap: 
```python
if opt.pre_img:
    self.pre_img_layer = nn.Sequential(
                            nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True))
if opt.pre_hm:
    self.pre_hm_layer = nn.Sequential(
                            nn.Conv2d(1, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                            nn.BatchNorm2d(channels[0], momentum=BN_MOMENTUM),
                            nn.ReLU(inplace=True))
```
Importantly, note that these are all defined exactly the same way with the exception of the input channels in `DLA.pre_hm_layer`.

To determine the analgous layers for `HarDNet`, we need to look at `HarDNetSeg.base = HarDNetBase(num_layers).base`. In `HarDNetBase.__init__()` we have the following relavant lines:
```python
class HarDNetBase(nn.Module):
    def __init__(self, arch, depth_wise=False):
        super().__init__()
        if arch == 85:
            first_ch  = [48, 96]
            second_kernel = 3

            ch_list = [  192, 256, 320, 480, 720]
            grmul = 1.7
            gr       = [  24, 24, 28, 36, 48]
            n_layers = [   8, 16, 16, 16, 16]
        elif arch == 68:
            first_ch  = [32, 64]
            second_kernel = 3

            ch_list = [  128, 256, 320, 640]
            grmul = 1.7
            gr       = [  14, 16, 20, 40]
            n_layers = [   8, 16, 16, 16]
        else:
            print("Error: HarDNet",arch," has no implementation.")
            exit()

        blks = len(n_layers)
        self.base = nn.ModuleList([])

        # First Layer: Standard Conv3x3, Stride=2
        self.base.append(ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2,  bias=False))
```
The important line is the last one defining the very first layer in the entire model. This structure is a custom class that produces a `Conv2d`, `BatchNorm2d`, and `ReLU` activation. Inserting a print statement after the last line above shows us its structure:
```
HarDNetBase.base[0]:
ConvLayer(
  (conv): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (norm): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
)
```
Note that the `48` for the output channels is half the first entry in the `channels` variable in `HarDNetSeg` (`98` for `num_layers=85` and `64` for `num_layers=68`). This means we can create new layers `HarDNetSeg.base_layer`, `HarDNetSeg.pre_img_layer`, and `HarDNetSeg.pre_hm_layer` by copying the above line that creates the first layer in `HarDNetBase` (only adjusting the number for `in_channels=1` for the `pre_hm_layer`):
```python
# NEW lines for the base_layer and previous image and heatmap
self.base_layer = ConvLayer(in_channels=3, out_channels=int(channels[0]/2), kernel=3, stride=2,  bias=False)
print(f'HarDNetSeg self.base_layer:')
print(self.base_layer)
if opt.pre_img:
    self.pre_img_layer = ConvLayer(in_channels=3, out_channels=int(channels[0]/2), kernel=3, stride=2,  bias=False)
    print(f'HarDNetSeg self.pre_img_layer:')
    print(self.pre_img_layer)
if opt.pre_hm:
    self.pre_hm_layer = ConvLayer(in_channels=1, out_channels=int(channels[0]/2), kernel=3, stride=2,  bias=False)
    print(f'HarDNetSeg self.pre_hm_layer:')
    print(self.pre_hm_layer)
```
The printout for the `base_layer` is 
```
HarDNetSeg self.base_layer:
ConvLayer(
  (conv): Conv2d(3, 48, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
  (norm): BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
)
```
Note it is identical to the printout for `HarDNetBase.base[0]` above. Now we are ready to investigate the forward pass of the `HarDNetSeg.forward`. First we modify the method's signature to include keyword arguments for `pre_img=None, pre_hm=None` to run the model without getting an error. We insert print statements to see the input tensor shape as it progresses through the network:
```python
def forward(self, x, pre_img=None, pre_hm=None):
    print(f'\n============== In hardnet.py HarDNetSeg.forward: ==============\n')
    xs = []
    x_sc = []
    
    print(f'Before self.base for loop x.shape: {x.shape}')
    for i in range(len(self.base)):
        x = self.base[i](x)
        print(f'HarDNetSeg self.base[{i}](x).shape: {x.shape}')
        if i in self.skip_nodes:
            print(f'Skip connection added at i={i}')
            xs.append(x)
    
    x = self.last_proj(x)
    x = self.last_pool(x)
    x2 = self.avg9x9(x)
    x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
    x = torch.cat([x,x2,x3],1)
    x = self.last_blk(x)
    
    for i in range(3):
        skip_x = xs[3-i]
        x = self.transUpBlocks[i](x, skip_x, (i<self.skip_lv))
        x = self.conv1x1_up[i](x)
        if self.SC[i] > 0:
            end = x.shape[1]
            x_sc.append( x[:,end-self.SC[i]:,:,:].contiguous() )
            x = x[:,:end-self.SC[i],:,:].contiguous()
        x2 = self.avg9x9(x)
        x3 = x/(x.sum((2,3),keepdim=True) + 0.1)
        x = torch.cat([x,x2,x3],1)
        x = self.denseBlocksUp[i](x)

    scs = [x]
    for i in range(3):
        if self.SC[i] > 0:
        scs.insert(0, F.interpolate(x_sc[i], size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True))

    x = torch.cat(scs,1)
    print(f'\n(x = torch.cat(scs,1)).shape: {x.shape}\n')
    z = {}
    for head in self.heads:
        z[head] = self.__getattr__(head)(x)
        print(f'head: {head}')
		print(f'z[head].shape: {z[head].shape}')
    
    print(f'\n============== End of HarDNetSeg forward pass =================\n')

    if self.trt:
        return [z[h] for h in z]
    return [z]
```
This produces the following printout:
```
============== In hardnet.py HarDNetSeg.forward: ==============

Before self.base for loop x.shape: torch.Size([1, 3, 608, 704])
HarDNetSeg self.base[0](x).shape: torch.Size([1, 48, 304, 352])
HarDNetSeg self.base[1](x).shape: torch.Size([1, 96, 304, 352])
Skip connection added at i=1
HarDNetSeg self.base[2](x).shape: torch.Size([1, 96, 152, 176])
HarDNetSeg self.base[3](x).shape: torch.Size([1, 214, 152, 176])
Skip connection added at i=3
HarDNetSeg self.base[4](x).shape: torch.Size([1, 192, 152, 176])
HarDNetSeg self.base[5](x).shape: torch.Size([1, 192, 76, 88])
HarDNetSeg self.base[6](x).shape: torch.Size([1, 392, 76, 88])
HarDNetSeg self.base[7](x).shape: torch.Size([1, 256, 76, 88])
HarDNetSeg self.base[8](x).shape: torch.Size([1, 458, 76, 88])
Skip connection added at i=8
HarDNetSeg self.base[9](x).shape: torch.Size([1, 320, 76, 88])
HarDNetSeg self.base[10](x).shape: torch.Size([1, 320, 38, 44])
HarDNetSeg self.base[11](x).shape: torch.Size([1, 588, 38, 44])
HarDNetSeg self.base[12](x).shape: torch.Size([1, 480, 38, 44])
HarDNetSeg self.base[13](x).shape: torch.Size([1, 784, 38, 44])
Skip connection added at i=13

(x = torch.cat(scs,1)).shape: torch.Size([1, 200, 152, 176])

head: hm
z[head].shape: torch.Size([1, 80, 152, 176])
head: reg
z[head].shape: torch.Size([1, 2, 152, 176])
head: wh
z[head].shape: torch.Size([1, 2, 152, 176])
head: tracking
z[head].shape: torch.Size([1, 2, 152, 176])

============== End of HarDNetSeg forward pass =================
```
If we print out the result given from `Detector.run`, it looks like
```
{'score': 0.49735987, 'class': 3, 'ct': array([171.81818, 132.09091], dtype=float32), 'tracking': array([ 3.877182 , -2.4245605], dtype=float32), 'bbox': array([170.9446 , 130.65312, 174.35144, 133.23135], dtype=float32), 'tracking_id': 1, 'age': 1, 'active': 1}
```
Compare this to the output with DLA:
```
{'score': 0.6022488, 'class': 3, 'ct': array([171.81818, 132.09091], dtype=float32), 'tracking': array([ 1.8437347, -2.1887512], dtype=float32), 'bbox': array([149.76167, 119.34333, 196.64006, 146.76524], dtype=float32), 'tracking_id': 1, 'age': 1, 'active': 1}
```
It's close to the same output!

What remains is to combine the previous image and heatmap with the current image (after they are run through their respective layers). Recall in `DLA.forward`, this was done as
```python
x = self.base_layer(x)
if pre_img is not None:
    x = x + self.pre_img_layer(pre_img)
if pre_hm is not None:
    x = x + self.pre_hm_layer(pre_hm)
```
We can copy this same code, only needing to adjust the initial `for` loop's starting index from `0` to `1`:
```python
for i in range(1, len(self.base)):
    x = self.base[i](x)
```
This is because `HarDNetSeg.base_layer`, `HarDNetSeg.pre_img_layer`, and `HarDNetSeg.pre_hm_layer`, were based off of `HarDNetBase.base[0]`. This should not mess up the skip connections, since as evidenced in our printout, the first one happens at `i=1`. We add the following lines at the beginning of the `HarDNetSeg.forward` method:
```python
print(f'x.shape: {x.shape}')
x = self.base_layer(x)
print(f'HarDNetSeg self.base_layer(x).shape: {x.shape}\n')
if pre_img is not None:
    print(f'pre_img.shape: {pre_img.shape}')
    print(f'HarDNetSeg self.pre_img_layer(pre_img).shape: {self.pre_img_layer(pre_img).shape}\n')
    x = x + self.pre_img_layer(pre_img)
if pre_hm is not None:
    print(f'pre_hm.shape: {pre_hm.shape}')
    print(f'HarDNetSeg self.pre_hm_layer(pre_hm).shape: {self.pre_hm_layer(pre_hm).shape}\n')
    x = x + self.pre_hm_layer(pre_hm)
```
Now we have the printout:
```
============== In hardnet.py HarDNetSeg.forward: ==============

x.shape: torch.Size([1, 3, 608, 704])
HarDNetSeg self.base_layer(x).shape: torch.Size([1, 48, 304, 352])

pre_img.shape: torch.Size([1, 3, 608, 704])
HarDNetSeg self.pre_img_layer(pre_img).shape: torch.Size([1, 48, 304, 352])

Entering HarDNetSeg.forward for loop after combination with pre_img_layer/pre_hm_layer:
Before self.base for loop x.shape: torch.Size([1, 48, 304, 352])
HarDNetSeg self.base[1](x).shape: torch.Size([1, 96, 304, 352])
Skip connection added at i=1
HarDNetSeg self.base[2](x).shape: torch.Size([1, 96, 152, 176])
HarDNetSeg self.base[3](x).shape: torch.Size([1, 214, 152, 176])
Skip connection added at i=3
HarDNetSeg self.base[4](x).shape: torch.Size([1, 192, 152, 176])
HarDNetSeg self.base[5](x).shape: torch.Size([1, 192, 76, 88])
HarDNetSeg self.base[6](x).shape: torch.Size([1, 392, 76, 88])
HarDNetSeg self.base[7](x).shape: torch.Size([1, 256, 76, 88])
HarDNetSeg self.base[8](x).shape: torch.Size([1, 458, 76, 88])
Skip connection added at i=8
HarDNetSeg self.base[9](x).shape: torch.Size([1, 320, 76, 88])
HarDNetSeg self.base[10](x).shape: torch.Size([1, 320, 38, 44])
HarDNetSeg self.base[11](x).shape: torch.Size([1, 588, 38, 44])
HarDNetSeg self.base[12](x).shape: torch.Size([1, 480, 38, 44])
HarDNetSeg self.base[13](x).shape: torch.Size([1, 784, 38, 44])
Skip connection added at i=13

(x = torch.cat(scs,1)).shape: torch.Size([1, 200, 152, 176])

head: hm
z[head].shape: torch.Size([1, 80, 152, 176])
head: reg
z[head].shape: torch.Size([1, 2, 152, 176])
head: wh
z[head].shape: torch.Size([1, 2, 152, 176])
head: tracking
z[head].shape: torch.Size([1, 2, 152, 176])

============== End of HarDNetSeg forward pass =================
```
Notice that the output head tensor shapes are the same that we had with `DLASeg` produced in its network head that was defined in `BaseModel`. This is further validation that we can safely ignore the `BaseModel` class for our HarDNet implementation. Note further that the input resolution is only downsampled by 2 `4` times instead of `5` as in the DLA model. This means that we can try to set the flags set for `--input_w(h)` to be what our actual input image resolution will be. Doing this does not result in any errors and we see a similar output:
```
============== In hardnet.py HarDNetSeg.forward: ==============

x.shape: torch.Size([1, 3, 616, 720])
HarDNetSeg self.base_layer(x).shape: torch.Size([1, 48, 308, 360])

pre_img.shape: torch.Size([1, 3, 616, 720])
HarDNetSeg self.pre_img_layer(pre_img).shape: torch.Size([1, 48, 308, 360])

Entering HarDNetSeg.forward for loop after combination with pre_img_layer/pre_hm_layer:
Before self.base for loop x.shape: torch.Size([1, 48, 308, 360])
HarDNetSeg self.base[1](x).shape: torch.Size([1, 96, 308, 360])
Skip connection added at i=1
HarDNetSeg self.base[2](x).shape: torch.Size([1, 96, 154, 180])
HarDNetSeg self.base[3](x).shape: torch.Size([1, 214, 154, 180])
Skip connection added at i=3
HarDNetSeg self.base[4](x).shape: torch.Size([1, 192, 154, 180])
HarDNetSeg self.base[5](x).shape: torch.Size([1, 192, 77, 90])
HarDNetSeg self.base[6](x).shape: torch.Size([1, 392, 77, 90])
HarDNetSeg self.base[7](x).shape: torch.Size([1, 256, 77, 90])
HarDNetSeg self.base[8](x).shape: torch.Size([1, 458, 77, 90])
Skip connection added at i=8
HarDNetSeg self.base[9](x).shape: torch.Size([1, 320, 77, 90])
HarDNetSeg self.base[10](x).shape: torch.Size([1, 320, 38, 45])
HarDNetSeg self.base[11](x).shape: torch.Size([1, 588, 38, 45])
HarDNetSeg self.base[12](x).shape: torch.Size([1, 480, 38, 45])
HarDNetSeg self.base[13](x).shape: torch.Size([1, 784, 38, 45])
Skip connection added at i=13

(x = torch.cat(scs,1)).shape: torch.Size([1, 200, 154, 180])

head: hm
z[head].shape: torch.Size([1, 80, 154, 180])
head: reg
z[head].shape: torch.Size([1, 2, 154, 180])
head: wh
z[head].shape: torch.Size([1, 2, 154, 180])
head: tracking
z[head].shape: torch.Size([1, 2, 154, 180])

============== End of HarDNetSeg forward pass =================
```
Note that the input tensor size is now `x.shape: torch.Size([1, 3, 616, 720])` for an image resolution of `720` pixels in width by `616` pixels in height.

Currently, the network does not detect anything with this construction, because `HarDNetSeg.base_layer`, `HarDNetSeg.pre_img_layer`, and `HarDNetSeg.pre_hm_layer` have no pre-trained weights. However, we can test out using `HarDNetSeg.base[0]` instead, since this layer does have pretrained weights. Testing that out with the following lines at the top of the `HarDNetSeg.forward` method:
print(f'x.shape: {x.shape}')
```python
x = self.base[0](x)
print(f'HarDNetSeg self.base[0](x).shape: {x.shape}\n')
if pre_img is not None:
    print(f'pre_img.shape: {pre_img.shape}')
    print(f'HarDNetSeg self.base[0](pre_img).shape: {self.base[0](pre_img).shape}\n')
    x = x + self.pre_img_layer(pre_img)
if pre_hm is not None:
    print(f'pre_hm.shape: {pre_hm.shape}')
    print(f'HarDNetSeg self.base[0](pre_hm).shape: {self.base[0](pre_hm).shape}\n')
    x = x + self.pre_hm_layer(pre_hm)
```
gives us the following printout where we see the correct tensor shapes:
```
============== In hardnet.py HarDNetSeg.forward: ==============

x.shape: torch.Size([1, 3, 608, 704])
HarDNetSeg self.base[0](x).shape: torch.Size([1, 48, 304, 352])

pre_img.shape: torch.Size([1, 3, 608, 704])
HarDNetSeg self.base[0](pre_img).shape: torch.Size([1, 48, 304, 352])
```
and we even get a decent detection:
```
{'score': 0.6265698, 'class': 3, 'ct': array([171.81818, 132.09091], dtype=float32), 'tracking': array([3.0315247, 1.6415405], dtype=float32), 'bbox': array([173.08401, 129.87735, 177.68637, 129.87735], dtype=float32), 'tracking_id': 1, 'age': 1, 'active': 1}
```
which is very similar to what we got when we ignored the previous image info:
```
{'score': 0.49735987, 'class': 3, 'ct': array([171.81818, 132.09091], dtype=float32), 'tracking': array([ 3.877182 , -2.4245605], dtype=float32), 'bbox': array([170.9446 , 130.65312, 174.35144, 133.23135], dtype=float32), 'tracking_id': 1, 'age': 1, 'active': 1}
```
Now this was just for testing, and we will not use the pre-trained weights of `HarDNetSeg.base[0]` as a stand-in for `HarDNetSeg.base_layer` and `HarDNetSeg.pre_img_layer` but will train the model anew on the COCO dataset. This just further validates that we have constructed the correct forward pass that allows processing of the previous image and heatmap.

## Training `CenterTrack-HarDNet`

Now that we've incorporated the HarDNet network as the backbone of CenterTrack, its time to determine how to train the model. We begin by examining training inside the `CenterTrack` repo which has not been modified thus far. Then we consider what changes we may need to make inside `CenterTrack_mod` to allow training with the new backbone

### Training `CenterTrack`

CenterTrack includes an experiment for training with tracking on COCO. It is based on a pre-trained model from the CenterNet repository: `ctdet_coco_dla_2x.pth`. We download it and place it in `CenterTrack/models/`. Then we create a new directory for the data, `CenterTrack/data` and make symbolic links to our downloaded COCO 2017 data:
```
ln -s /path/to/downloaded/coco2017 /path/to/CenterTrack/data/coco
``` 

The command to start the training experiment should be run in `CenterTrack/src`:
```
python3 main.py tracking --exp_id coco_tracking --tracking --load_model ../../CenterNet/models/ctdet_coco_dla_2x.pth  --gpus 0 --batch_size 8 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
```
This command initially produced an error:
```
subprocess.CalledProcessError: Command '['git', 'describe']' returned non-zero exit status 128.
```
which was the subject of an [issue on the CenterTrack Repo](https://github.com/xingyizhou/CenterTrack/issues/98). The solution is to modify `CenterTrack/src/lib/logger.py` line 33 from `subprocess.check_output(["git", "describe"])))` to `subprocess.check_output(["git", "describe", "--always"])))`. After this modification, the training sucessfully started and produced the following output:
```
jugaadlabs@bionic:~/JL/centertrack_testing/CenterTrack/src$ python3 main.py tracking --exp_id coco_tracking --tracking --load_model ../../CenterNet/models/ctdet_coco_dla_2x.pth  --gpus 0 --batch_size 8 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
Running tracking
Using tracking threshold for out threshold! 0.3
Fix size testing.
training chunk_sizes: [8]
input h w: 512 512
heads {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}
weights {'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}
head conv {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
Namespace(K=100, add_05=False, amodel_offset_weight=1, arch='dla_34', aug_rot=0, backbone='dla34', batch_size=8, chunk_sizes=[8], custom_dataset_ann_path='', custom_dataset_img_path='', data_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../data', dataset='coco', dataset_version='', debug=0, debug_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../exp/tracking/coco_tracking/debug', debugger_theme='white', demo='', dense_reg=1, dep_weight=1, depth_scale=1, dim_weight=1, dla_node='dcn', down_ratio=4, efficient_level=0, eval_val=False, exp_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../exp/tracking', exp_id='coco_tracking', fix_res=True, fix_short=-1, flip=0.5, flip_test=False, fp_disturb=0.1, gpus=[0], gpus_str='0', head_conv={'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}, head_kernel=3, heads={'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}, hm_disturb=0.05, hm_hp_weight=1, hm_weight=1, hp_weight=1, hungarian=False, ignore_loaded_cats=[], input_h=512, input_res=512, input_w=512, keep_res=False, kitti_split='3dop', load_model='../../CenterNet/models/ctdet_coco_dla_2x.pth', load_results='', lost_disturb=0.4, lr=0.0005, lr_step=[60], ltrb=False, ltrb_amodal=False, ltrb_amodal_weight=0.1, ltrb_weight=0.1, map_argoverse_id=False, master_batch_size=8, max_age=-1, max_frame_dist=3, model_output_list=False, msra_outchannel=256, neck='dlaup', new_thresh=0.3, nms=False, no_color_aug=False, no_pause=False, no_pre_img=False, non_block_test=False, not_cuda_benchmark=False, not_idaup=False, not_max_crop=False, not_prefetch_test=False, not_rand_crop=False, not_set_cuda_env=False, not_show_bbox=False, not_show_number=False, not_show_txt=False, num_classes=80, num_epochs=70, num_head_conv=1, num_iters=-1, num_layers=101, num_stacks=1, num_workers=16, nuscenes_att=False, nuscenes_att_weight=1, off_weight=1, only_show_dots=False, optim='adam', out_thresh=0.3, output_h=128, output_res=128, output_w=128, pad=31, pre_hm=True, pre_img=True, pre_thresh=0.3, print_iter=0, prior_bias=-4.6, public_det=False, qualitative=False, reg_loss='l1', reset_hm=False, resize_video=False, resume=False, reuse_hm=False, root_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../..', rot_weight=1, rotate=0, same_aug_pre=False, save_all=False, save_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../exp/tracking/coco_tracking', save_framerate=30, save_img_suffix='', save_imgs=[], save_point=[90], save_results=False, save_video=False, scale=0.05, seed=317, shift=0.05, show_trace=False, show_track_color=False, skip_first=-1, tango_color=False, task='tracking', test=False, test_dataset='coco', test_focal_length=-1, test_scales=[1.0], track_thresh=0.3, tracking=True, tracking_weight=1, trainval=False, transpose_video=False, use_kpt_center=False, use_loaded_results=False, val_intervals=10000, velocity=False, velocity_weight=1, video_h=512, video_w=512, vis_gt_bev='', vis_thresh=0.3, weights={'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}, wh_weight=0.1, zero_pre_hm=False, zero_tracking=False)
Creating model...
Using node type: (<class 'model.networks.dla.DeformConv'>, <class 'model.networks.dla.DeformConv'>)
Warning: No ImageNet pretrain!!
loaded ../../CenterNet/models/ctdet_coco_dla_2x.pth, epoch 230
Drop parameter base.fc.weight.
Drop parameter base.fc.bias.
No param tracking.0.weight.
No param tracking.0.bias.
No param tracking.2.weight.
No param tracking.2.bias.
No param base.pre_img_layer.0.weight.
No param base.pre_img_layer.1.weight.
No param base.pre_img_layer.1.bias.
No param base.pre_img_layer.1.running_mean.
No param base.pre_img_layer.1.running_var.
No param base.pre_img_layer.1.num_batches_tracked.
No param base.pre_hm_layer.0.weight.
No param base.pre_hm_layer.1.weight.
No param base.pre_hm_layer.1.bias.
No param base.pre_hm_layer.1.running_mean.
No param base.pre_hm_layer.1.running_var.
No param base.pre_hm_layer.1.num_batches_tracked.
Setting up train data...
==> initializing train data from /home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../data/coco/annotations/instances_train2017.json, 
 images from /home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../data/coco/train2017 ...
loading annotations into memory...
Done (t=9.01s)
creating index...
index created!
Creating video index!
Loaded train 118287 samples
Starting training...
tracking/coco_tracking |                                | train: [1][8/14785]|Tot: 0:00:09 |ETA: 4:45:32 |tot 8.1381 |hm 2.8005 |wh 4.8889 |reg 0.2328 |tracking 4.6159 |Data 0.001s(0.195s) |Net 1.104s
```
The batch size was adjusted from the original command of `128` all the way down to `8` so as to not run out of memory.

Now this just verifies that the `CenterTrack` repository is working as intended (albeit with a small modification of one file). We now look to determine how to train a model without relying on a pre-trained model from CenterNet. We try this by removing the `--load_model` flag from the command:
```
python3 main.py tracking --exp_id coco_tracking --tracking  --gpus 0 --batch_size 8 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
```
This also works and produces the below output:
```
jugaadlabs@bionic:~/JL/centertrack_testing/CenterTrack/src$ python3 main.py tracking --exp_id coco_tracking --tracking  --gpus 0 --batch_size 8 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
Running tracking
Using tracking threshold for out threshold! 0.3
Fix size testing.
training chunk_sizes: [8]
input h w: 512 512
heads {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}
weights {'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}
head conv {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
Namespace(K=100, add_05=False, amodel_offset_weight=1, arch='dla_34', aug_rot=0, backbone='dla34', batch_size=8, chunk_sizes=[8], custom_dataset_ann_path='', custom_dataset_img_path='', data_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../data', dataset='coco', dataset_version='', debug=0, debug_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../exp/tracking/coco_tracking/debug', debugger_theme='white', demo='', dense_reg=1, dep_weight=1, depth_scale=1, dim_weight=1, dla_node='dcn', down_ratio=4, efficient_level=0, eval_val=False, exp_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../exp/tracking', exp_id='coco_tracking', fix_res=True, fix_short=-1, flip=0.5, flip_test=False, fp_disturb=0.1, gpus=[0], gpus_str='0', head_conv={'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}, head_kernel=3, heads={'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}, hm_disturb=0.05, hm_hp_weight=1, hm_weight=1, hp_weight=1, hungarian=False, ignore_loaded_cats=[], input_h=512, input_res=512, input_w=512, keep_res=False, kitti_split='3dop', load_model='', load_results='', lost_disturb=0.4, lr=0.0005, lr_step=[60], ltrb=False, ltrb_amodal=False, ltrb_amodal_weight=0.1, ltrb_weight=0.1, map_argoverse_id=False, master_batch_size=8, max_age=-1, max_frame_dist=3, model_output_list=False, msra_outchannel=256, neck='dlaup', new_thresh=0.3, nms=False, no_color_aug=False, no_pause=False, no_pre_img=False, non_block_test=False, not_cuda_benchmark=False, not_idaup=False, not_max_crop=False, not_prefetch_test=False, not_rand_crop=False, not_set_cuda_env=False, not_show_bbox=False, not_show_number=False, not_show_txt=False, num_classes=80, num_epochs=70, num_head_conv=1, num_iters=-1, num_layers=101, num_stacks=1, num_workers=16, nuscenes_att=False, nuscenes_att_weight=1, off_weight=1, only_show_dots=False, optim='adam', out_thresh=0.3, output_h=128, output_res=128, output_w=128, pad=31, pre_hm=True, pre_img=True, pre_thresh=0.3, print_iter=0, prior_bias=-4.6, public_det=False, qualitative=False, reg_loss='l1', reset_hm=False, resize_video=False, resume=False, reuse_hm=False, root_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../..', rot_weight=1, rotate=0, same_aug_pre=False, save_all=False, save_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../exp/tracking/coco_tracking', save_framerate=30, save_img_suffix='', save_imgs=[], save_point=[90], save_results=False, save_video=False, scale=0.05, seed=317, shift=0.05, show_trace=False, show_track_color=False, skip_first=-1, tango_color=False, task='tracking', test=False, test_dataset='coco', test_focal_length=-1, test_scales=[1.0], track_thresh=0.3, tracking=True, tracking_weight=1, trainval=False, transpose_video=False, use_kpt_center=False, use_loaded_results=False, val_intervals=10000, velocity=False, velocity_weight=1, video_h=512, video_w=512, vis_gt_bev='', vis_thresh=0.3, weights={'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}, wh_weight=0.1, zero_pre_hm=False, zero_tracking=False)
Creating model...
Using node type: (<class 'model.networks.dla.DeformConv'>, <class 'model.networks.dla.DeformConv'>)
Setting up train data...
==> initializing train data from /home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../data/coco/annotations/instances_train2017.json, 
 images from /home/jugaadlabs/JL/centertrack_testing/CenterTrack/src/lib/../../data/coco/train2017 ...
loading annotations into memory...
Done (t=8.79s)
creating index...
index created!
Creating video index!
Loaded train 118287 samples
Starting training...
tracking/coco_tracking |                                | train: [1][4/14785]|Tot: 0:00:07 |ETA: 6:36:22 |tot 11.7923 |hm 5.3122 |wh 22.3037 |reg 0.3743 |tracking 3.8754 |Data 0.001s(0.456s) |Net 1.415s
```
Note there is now no printouts of the form `Drop parameter base.fc.bias.` or `No param tracking.0.weight.` which come from loading a `pytorch` model file and matching parameter names. We try one more modification by adding the `--arch` flag to specify a non-default model architecture:
```
python3 main.py tracking --exp_id coco_tracking --tracking --arch dla_169 --gpus 0 --batch_size 2 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
```
We needed to drop the batch size down to `2` to avoid memory issues. This is promising becuase we should be able to do the same command, but specify our HarDNet backbone with the `--arch` flag.

## Training CenterTrack-HarDNet

We first verify the above sample training method also works in `CenterTrack_mod` (after making the same change in `CenterTrack_mod/src/lib/logger.py`). So far none of our changes in `CenterTrack_mod` has affected the ability to perform the example training command:
```
python3 main.py tracking --exp_id coco_tracking --tracking --load_model ../../CenterNet/models/ctdet_coco_dla_2x.pth  --gpus 0 --batch_size 8 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
```
This produced the same result as in the `CenterTrack`. We move on to what we think will be the correct training command for HarDNet:
```
python3 main.py tracking --exp_id coco_tracking --tracking --arch hardnet_85 --head_conv 256 --input_w 720 --input_h 616 --gpus 0 --batch_size 1 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
```
The batch size needed to be further reduced to not run out of memory. After removing the extra print statements added in our study above, the printout looks promising:
```
jugaadlabs@bionic:~/JL/centertrack_testing/CenterTrack_mod/src$ python3 main.py tracking --exp_id coco_tracking --tracking --arch hardnet_85 --head_conv 256 --input_w 720 --input_h 616 --gpus 0 --batch_size 1 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1
Running tracking
Using tracking threshold for out threshold! 0.3
Fix size testing.
training chunk_sizes: [1]
input h w: 616 720
heads {'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}
weights {'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}
head conv {'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}
Namespace(K=100, add_05=False, amodel_offset_weight=1, arch='hardnet_85', aug_rot=0, backbone='dla34', batch_size=1, chunk_sizes=[1], custom_dataset_ann_path='', custom_dataset_img_path='', data_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../../data', dataset='coco', dataset_version='', debug=0, debug_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../../exp/tracking/coco_tracking/debug', debugger_theme='white', demo='', dense_reg=1, dep_weight=1, depth_scale=1, dim_weight=1, dla_node='dcn', down_ratio=4, efficient_level=0, eval_val=False, exp_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../../exp/tracking', exp_id='coco_tracking', fix_res=True, fix_short=-1, flip=0.5, flip_test=False, fp_disturb=0.1, gpus=[0], gpus_str='0', head_conv={'hm': [256], 'reg': [256], 'wh': [256], 'tracking': [256]}, head_kernel=3, heads={'hm': 80, 'reg': 2, 'wh': 2, 'tracking': 2}, hm_disturb=0.05, hm_hp_weight=1, hm_weight=1, hp_weight=1, hungarian=False, ignore_loaded_cats=[], input_h=616, input_res=720, input_w=720, keep_res=False, kitti_split='3dop', load_model='', load_results='', lost_disturb=0.4, lr=0.0005, lr_step=[60], ltrb=False, ltrb_amodal=False, ltrb_amodal_weight=0.1, ltrb_weight=0.1, map_argoverse_id=False, master_batch_size=1, max_age=-1, max_frame_dist=3, model_output_list=False, msra_outchannel=256, neck='dlaup', new_thresh=0.3, nms=False, no_color_aug=False, no_pause=False, no_pre_img=False, non_block_test=False, not_cuda_benchmark=False, not_idaup=False, not_max_crop=False, not_prefetch_test=False, not_rand_crop=False, not_set_cuda_env=False, not_show_bbox=False, not_show_number=False, not_show_txt=False, num_classes=80, num_epochs=70, num_head_conv=1, num_iters=-1, num_layers=101, num_stacks=1, num_workers=16, nuscenes_att=False, nuscenes_att_weight=1, off_weight=1, only_show_dots=False, optim='adam', out_thresh=0.3, output_h=154, output_res=180, output_w=180, pad=31, pre_hm=True, pre_img=True, pre_thresh=0.3, print_iter=0, prior_bias=-4.6, public_det=False, qualitative=False, reg_loss='l1', reset_hm=False, resize_video=False, resume=False, reuse_hm=False, root_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../..', rot_weight=1, rotate=0, same_aug_pre=False, save_all=False, save_dir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../../exp/tracking/coco_tracking', save_framerate=30, save_img_suffix='', save_imgs=[], save_point=[90], save_results=False, save_video=False, scale=0.05, seed=317, shift=0.05, show_trace=False, show_track_color=False, skip_first=-1, tango_color=False, task='tracking', test=False, test_dataset='coco', test_focal_length=-1, test_scales=[1.0], track_thresh=0.3, tracking=True, tracking_weight=1, trainval=False, transpose_video=False, use_kpt_center=False, use_loaded_results=False, val_intervals=10000, velocity=False, velocity_weight=1, video_h=512, video_w=512, vis_gt_bev='', vis_thresh=0.3, weights={'hm': 1, 'reg': 1, 'wh': 0.1, 'tracking': 1}, wh_weight=0.1, zero_pre_hm=False, zero_tracking=False)
Creating model...
3 x 3 x 3 x 48
3 x 3 x 48 x 96
3 x 3 x 96 x 24
3 x 3 x 120 x 40
3 x 3 x 40 x 24
3 x 3 x 160 x 70
3 x 3 x 70 x 24
3 x 3 x 94 x 40
3 x 3 x 40 x 24
3 x 3 x 230 x 118
Blk out = 214
1 x 1 x 214 x 192
3 x 3 x 192 x 24
3 x 3 x 216 x 40
3 x 3 x 40 x 24
3 x 3 x 256 x 70
3 x 3 x 70 x 24
3 x 3 x 94 x 40
3 x 3 x 40 x 24
3 x 3 x 326 x 118
3 x 3 x 118 x 24
3 x 3 x 142 x 40
3 x 3 x 40 x 24
3 x 3 x 182 x 70
3 x 3 x 70 x 24
3 x 3 x 94 x 40
3 x 3 x 40 x 24
3 x 3 x 444 x 200
Blk out = 392
1 x 1 x 392 x 256
3 x 3 x 256 x 28
3 x 3 x 284 x 48
3 x 3 x 48 x 28
3 x 3 x 332 x 80
3 x 3 x 80 x 28
3 x 3 x 108 x 48
3 x 3 x 48 x 28
3 x 3 x 412 x 138
3 x 3 x 138 x 28
3 x 3 x 166 x 48
3 x 3 x 48 x 28
3 x 3 x 214 x 80
3 x 3 x 80 x 28
3 x 3 x 108 x 48
3 x 3 x 48 x 28
3 x 3 x 550 x 234
Blk out = 458
1 x 1 x 458 x 320
3 x 3 x 320 x 36
3 x 3 x 356 x 62
3 x 3 x 62 x 36
3 x 3 x 418 x 104
3 x 3 x 104 x 36
3 x 3 x 140 x 62
3 x 3 x 62 x 36
3 x 3 x 522 x 176
3 x 3 x 176 x 36
3 x 3 x 212 x 62
3 x 3 x 62 x 36
3 x 3 x 274 x 104
3 x 3 x 104 x 36
3 x 3 x 140 x 62
3 x 3 x 62 x 36
3 x 3 x 698 x 300
Blk out = 588
1 x 1 x 588 x 480
3 x 3 x 480 x 48
3 x 3 x 528 x 82
3 x 3 x 82 x 48
3 x 3 x 610 x 138
3 x 3 x 138 x 48
3 x 3 x 186 x 82
3 x 3 x 82 x 48
3 x 3 x 748 x 236
3 x 3 x 236 x 48
3 x 3 x 284 x 82
3 x 3 x 82 x 48
3 x 3 x 366 x 138
3 x 3 x 138 x 48
3 x 3 x 186 x 82
3 x 3 x 82 x 48
3 x 3 x 984 x 400
Blk out = 784
1 x 1 x 784 x 256
3 x 3 x 768 x 80
3 x 3 x 848 x 136
3 x 3 x 136 x 80
3 x 3 x 984 x 232
3 x 3 x 232 x 80
3 x 3 x 312 x 136
3 x 3 x 136 x 80
3 x 3 x 1216 x 394
Blk out = 714
HarDNet85 Base Model loaded.
1 x 1 x 1498 x 256
3 x 3 x 672 x 64
3 x 3 x 736 x 108
3 x 3 x 108 x 64
3 x 3 x 844 x 184
3 x 3 x 184 x 64
3 x 3 x 248 x 108
3 x 3 x 108 x 64
3 x 3 x 1028 x 314
Blk out = 570
1 x 1 x 1028 x 192
3 x 3 x 480 x 48
3 x 3 x 528 x 82
3 x 3 x 82 x 48
3 x 3 x 610 x 138
3 x 3 x 138 x 48
3 x 3 x 186 x 82
3 x 3 x 82 x 48
3 x 3 x 748 x 236
Blk out = 428
1 x 1 x 642 x 96
3 x 3 x 288 x 28
3 x 3 x 316 x 48
3 x 3 x 48 x 28
3 x 3 x 364 x 80
Blk out = 136
3 x 3 x 3 x 48
3 x 3 x 3 x 48
3 x 3 x 1 x 48
Setting up train data...
==> initializing train data from /home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../../data/coco/annotations/instances_train2017.json, 
 images from /home/jugaadlabs/JL/centertrack_testing/CenterTrack_mod/src/lib/../../data/coco/train2017 ...
loading annotations into memory...
Done (t=9.40s)
creating index...
index created!
Creating video index!
Loaded train 118287 samples
Starting training...
tracking/coco_tracking |                                | train: [1][0/118287]|Tot: 0:00:05 |ETA: 0:00:00 |tot 3946.4241 |hm 3936.5435 |wh 69.6403 |reg 0.5919 |tracking 2.3244 |Data 0.755s(0.755s) |Net 5.tracking/coco_tracking |                                | train: [1][1/118287]|Tot: 0:00:06 |ETA: 8 days, 3:15:19 |tot 2157.1712 |hm 2146.9762 |wh 44.3326 |reg 0.6165 |tracking 5.1451 |Data 0.001s(0.378s)tracking/coco_tracking |                                | train: [1][2/118287]|Tot: 0:00:06 |ETA: 4 days, 4:38:42 |tot 1635.2595 |hm 1624.3816 |wh 49.4537 |reg 0.7200 |tracking 5.2125 |Data 0.002s(0.252s)tracking/coco_tracking |                                | train: [1][3/118287]|Tot: 0:00:06 |ETA: 2 days, 21:06:46 |tot 1562.1084 |hm 1549.5445 |wh 71.6997 |reg 0.7863 |tracking 4.6076 |Data 0.002s(0.190stracking/coco_tracking |                                | train: [1][4/118287]|Tot: 0:00:06 |ETA: 2 days, 5:18:42 |tot 1271.6334 |hm 1260.1652 |wh 58.5173 |reg 0.7560 |tracking 4.8604 |Data 0.002s(0.152s)tracking/coco_tracking |                                | train: [1][5/118287]|Tot: 0:00:06 |ETA: 1 day, 19:50:46 |tot 1068.5334 |hm 1057.1610 |wh 60.6321 |reg 0.7450 |tracking 4.5642 |Data 0.002s(0.127s)tracking/coco_tracking |                                | train: [1][6/118287]|Tot: 0:00:07 |ETA: 1 day, 13:32:03 |tot 933.1277 |hm 921.8911 |wh 52.7698 |reg 0.7189 |tracking 5.2407 |Data 0.003s(0.110s) |tracking/coco_tracking |                                | train: [1][7/118287]|Tot: 0:00:07 |ETA: 1 day, 9:00:51 |tot 820.8706 |hm 810.0787 |wh 47.4278 |reg 0.7162 |tracking 5.3328 |Data 0.002s(0.096s) |Ntracking/coco_tracking |                                | train: [1][8/118287]|Tot: 0:00:07 |ETA: 1 day, 5:38:50 |tot 737.5732 |hm 726.5328 |wh 45.5376 |reg 0.7499 |tracking 5.7367 |Data 0.002s(0.086s) |Net 0.822s 
```
We see the model is loaded with the correct input size and that training is begun at the end.


We test out the training script with the following command:
```
python3 main.py tracking --exp_id coco_tracking_720x616 --tracking --arch hardnet_85 --head_conv 256 --input_w 720 --input_h 616 --gpus 0 --batch_size 1 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --num_epochs 1
```

To better compare with the existing pre-trained model, we leave out the `--input_w(h)` so that input images are scaled down to 512x512:
```
python3 main.py tracking --exp_id coco_tracking --tracking --arch hardnet_85 --head_conv 256 --gpus 0 --batch_size 1 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --num_epochs 1
```


We train for a single epoch to determine if the script completes without an issue. After about 6 hours to run through all of the nearly 120k COCO2017 training images, get get a saved model file (named `model_last.pth`) and a log with the following summary:
```
2022-05-05-22-33: epoch: 1 |tot 10.607093 | hm 4.182101 | wh 17.019249 | reg 0.244340 | tracking 4.478727 | time 350.400000 | 
```
The last column is in minutes. If we trained for more epochs, this file would contain more lines showing the loss of each detection head progressively getting minimized (During training, the metrics are printed out on a new line for each batch and the metrics all decreased steadily before flattening out at the end). We can copy the model file into `CenterTrack_mod/models`, renaming it to something more descriptive with the partially trained weights and test it out with the demo script on a set of benchmarking images:
```
python3 demo.py tracking  --arch hardnet_85 --load_model ../models/coco_tracking_hardnet_85_720x616_1_epoch.pth --demo /home/jugaadlabs/JL/ML_benchmarking/benchmarking_images/parking_lot_driving/driveby_pass
```













































