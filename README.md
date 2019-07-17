# Mosaic creation through the stitching of aerial rectified images obtained from a moving vehicle

### Goal
The gol is mosaic creation through the stitching of aerial rectified images obtained from a moving vehicle.

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
$ python3 car_detection.py
$ python3 car_detection.py
$ python3 car_detection.py


```


### Requirements
| Software  | Version | Required|
| ------------- | ------------- |  ------------- |
| Python | >= 3.5  | Yes    |
| Numpy  | Tested on 1.13 |    Yes     |
| Matplotlib  | >= 1.0  | Yes   |
| opencv-python| == 3.4.2.17  | Yes
| opencv-contrib-python  | == 3.4.2.17  |Yes |
| os  | -  |Yes |


### To run

