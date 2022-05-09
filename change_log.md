# Simple summary of changes made

Details on how we arrived at the following changes are provided in these [notes](/adding_hardnet_backbone_notes.md).

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