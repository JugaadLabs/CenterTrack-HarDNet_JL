# Simple summary of changes made to the CenterTrack repository

This repository was forked from [CenterTrack](https://github.com/xingyizhou/CenterTrack) and modified to use the HarDNet backbone used in the [CenterNet-HarDNet](https://github.com/PingoLH/CenterNet-HarDNet) repository.

Details on how we arrived at the following changes are provided in these [notes](/adding_hardnet_backbone_notes.md).

In the following, `repo_root` refers to the root of this repository.

## Changes to the repository directory structure

1. Added a new directories `repo_root/models` and `repo_root/data` and included `.gitignore` files to not track anything inside
    - `models` is to store pre-trained `.pth` model files
    - `data` is to store symbolic links to the COCO2017 dataset

## Changes needed from updated modules and APIs

1. In `repo_root/src/lib/model/networks/backbones/mobilenet.py`:
    - Changed import of `load_state_dict_from_url`

2. In `repo_root/src/lib/model/utils`:
    - Changed import of `linear_assignment`


## Changes needed to incorporate the HarDNet backbone

1. In `repo_root/src/lib/model/model.py`:
    - Added import of `HarDNetSeg` and corresponding entry in `_network_factory`

2. In `repo_root/src/lib/model/networks`:
    - Copied file `hardnet.py` from the `CenterNet-HarDNet` repository located at `CenterNet-HarDNet/src/lib/models/networks/hardnet.py`
    - Fixed inconsistent indentation throughout file

3. In the new `repo_root/src/lib/model/networks/hardnet.py`:
    - Changed definition of `HarDNetSeg` class to accomodate its import in Step 1 above
    - Modified line in for loop defining the network head due to a variable name change from previous step
    - Added additional layers in `HarDNetSeg.__init__` for the input current and previous image, and previous heatmap
    - Added new lines at the beginning of `HarDNetSeg.forward` method to account for combining current and prior image information
    - Modified `for` loop in `HarDNetSeg.forward` to skip the first layer due to above necessary step in CenterTrack which modifies the first layer pass

4. In `repo_root/src`:
    - Copied pre-trained model files `hartnet68_base.pth` and `hartnet85_base.pth` from the `CenterNet-HarDNet` repository located at `CenterNet-HarDNet/src`

5. In `repo_root/models`:
    - Downloaded pretrained model `centernet_hardnet85_coco_608.pth` from the `CenterNet-HarDNet` repository

## Changes needed to train the network

1. In `repo_root/src/lin/logger.py`:
    - Added additional argument `--always` to subprocess call (this is not related to HarDNet, but fixed an error that arose in training)

## Training command:

So far, we have tested out the following training command, run inside `repo_root/src`:
```
$ python3 main.py tracking --exp_id coco_tracking --tracking --arch hardnet_85 --head_conv 256 --gpus 0 --batch_size 1 --lr 5e-4 --num_workers 16 --pre_hm --shift 0.05 --scale 0.05 --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --num_epochs 1
```

This is a modified version of the command found in the `CenterTrack/experiments/coco_tracking.sh` script that was used to train the DLA model we have been using in IRIS. We will test out other flags that can be set to generate validation metrics and save the model periodically or even after every epoch:
```
--save_point		# String of integers for epochs to save model at. Use this or:
--save_all			# Saves model to disk after each epoch
--eval_val			# True/False to evaluate model on validation set
--val_intervals x	# Epoch interval to run validation (when epoch is in save_point list, or epoch % val_intervals == 0)
```

For overfitting, it has been suggested by the developers of the CenterNet repository (on which CenterTrack was made by the same developers) to use validation AP over the typical loss, as the validation loss may not behave normally (decrease for a while and then increase, see [this issue](https://github.com/xingyizhou/CenterNet/issues/148) for example and more details)

## Changes needed to convert the model to TensorRT

Still in progress