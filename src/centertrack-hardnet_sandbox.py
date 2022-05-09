import _init_paths

import os
import sys
import torch
import cv2

from detector import Detector
from opts import opts

# Function called by camnetGPU and camnetCPU initilizations
def load_detector(centertrackdir='/home/jugaadlabs/JL/centertrack_testing/CenterTrack', img_width=720, img_height=616):
    args = [#'task=tracking',
           #'--arch=hardnet_85',
           #'--load_model={}'.format(os.path.join(centertrackdir, 'models', 'coco_tracking.pth')),
           '--input_w={}'.format(img_width),
           '--input_h={}'.format(img_height)]

    opt = opts().init(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    # print(f"opt.task: {opt.task}")
    detector = Detector(opt)

    return detector

# Function called by camnetGPU.runTarget
def run_detector_gpu(detector, image):
	"""
	Function to perform object detection on the GPU
	"""

	with torch.no_grad():
		ret = detector.run(image)

	# Return the 'results' value, which is a dictionary of the form 
	# {'score': x.xx, 
	#  'class': int, 
	#  'ct': array([xx, yy], dtype=float32), 
	#  'tracking': array([xx, yy], dtype=float32), 
	#  'bbox': array([x1, y1, x2, y2], dtype=float32), 
	#  'tracking_id': int, 
	#  'age': int (likely 1), 
	#  'active': int}
	return ret['results']


if __name__ == "__main__":
    sys.argv.append('task=tracking')
    # sys.argv.append('--input_w=720')
    # sys.argv.append('--input_h=616')
    # sys.argv.append('--input_w=704')
    # sys.argv.append('--input_h=608')

    # Un-comment for HarDNet-68:
    # sys.argv.append('--arch=hardnet_68')
    # Un-comment for HarDNet-85:
    sys.argv.append('--arch=hardnet_85')
    sys.argv.append('--head_conv=256')
    sys.argv.append('--load_model={}'.format(os.path.join('/home/jugaadlabs/JL/CenterTrack-HarDNet_JL', 'models', 'centernet_hardnet85_coco_608.pth')))
    


    # Un-comment for DLA-34:
    # sys.argv.append('--arch=dla_34')
    # sys.argv.append('--load_model={}'.format(os.path.join('/home/jugaadlabs/JL/centertrack_testing/CenterTrack', 'models', 'coco_tracking.pth')))



    opt = opts().init()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    print('===== opt properties =====')
    print(f"opt.gpus_str: {opt.gpus_str}")
    print(f"opt.arch: {opt.arch}")
    print(f"opt.task: {opt.task}")
    print(f"opt.heads: {opt.heads}")
    print(f"opt.head_conv: {opt.head_conv}")
    print(f"opt.input_h: {opt.input_h}")
    print(f"opt.input_w: {opt.input_w}")
    print('==========================')
    detector = Detector(opt)



    # sys.argv.append('--load_model={}'.format(os.path.join('/home/jugaadlabs/JL/centertrack_testing/CenterTrack', 'models', 'coco_tracking.pth')))
    images_path = '/home/jugaadlabs/JL/ML_benchmarking/benchmarking_images/parking_lot_driving/driveby_pass'
    image_filenames = [f'driveby_left_wide_frame_{x}.png' for x in range(1326, 1426)]

    image = cv2.imread(os.path.join(images_path, image_filenames[50]))
    # cv2.imshow('test', image)
    # cv2.waitKey()
    # cv2.destroyWindow('test')
    # print(f'image.shape: {image.shape}')
    # resized_image = cv2.resize(image, (704, 608))


    # # detector = load_detector(arch='dla_34', model='ctdet_coco_dla_1x.pth', img_width=704, img_height=608)
    # # detector = load_detector(arch='hourglass', model='ctdet_coco_hg.pth', img_width=704, img_height=608)
    # detector = load_detector()

    results = detector.run(image)

    if len(results['results']) > 0:
      for res in results['results']:
          print(res)
    else:
      print(results)