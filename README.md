# Mosaic creation through the stitching of aerial rectified images obtained from a moving vehicle

### Goal
The gol is mosaic creation through the stitching of aerial rectified images obtained from a moving vehicle.

### Dataset
Stuttgart of Kitti (https://cityscapes-dataset.com/downloads/), in particular leftImg8bit_demoVideo (https://drive.google.com/open?id=1GrhTcMOxjPN-Ei-k9IoUW1TAGNXe-ktn).
You can put the frames to processing into the folder dataset.
To compute correctly the rectified images, use the right camera info into the MATLAB script (camera.json for stuttgart https://drive.google.com/open?id=1tDMvRqau5c_hhBb1UioJgK9GP6kyJfgM).

### Pipeline
1. Cars detection
2. Rectification 
3. Keypoints detections with SURF
4. Compute matrix Homography
5. Image warping
6. Overlap to mosaic creation

### To run
```
$ python3 car_detection.py
```

To compute the rectified images run on MATLAB the script: compute_bird_view.m

```
$ python3 create_submosaic.py
$ python3 merge_submosaic.py
```

### Submosaics
#### Results
First submosaic               |  Second submosaic
:-------------------------:|:-------------------------:
![](https://github.com/AlessandroMinervini/Mosaic-creation-through-the-stitching-of-aerial-rectified-images-obtained-from-a-moving-vehicle/blob/master/readme_images/submosaic33.jpg) | ![](https://github.com/AlessandroMinervini/Mosaic-creation-through-the-stitching-of-aerial-rectified-images-obtained-from-a-moving-vehicle/blob/master/readme_images/submosaic66.jpg)

### Merge submosaics
#### Results: 
First submosaic + second submosaic
:-------------------------:
![](https://github.com/AlessandroMinervini/Mosaic-creation-through-the-stitching-of-aerial-rectified-images-obtained-from-a-moving-vehicle/blob/master/readme_images/merge_submosaic.jpg)

### Requirements
| Software  | Version | Required|
| ------------- | ------------- |  ------------- |
| Python | >= 3.5  | Yes    |
| Numpy  | Tested on 1.13 |    Yes     |
| Matplotlib  | >= 1.0  | Yes   |
| opencv-python| == 3.4.2.17  | Yes
| opencv-contrib-python  | == 3.4.2.17  |Yes |
| os  | -  |Yes |

