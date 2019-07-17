import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import *
from numpy import linalg

def merge_submosaic(img1, img2, mosaic1, mosaic2):
    MIN_MATCH_COUNT = 4
    threshold = 0.3

    height= mosaic2.shape[0] * 2
    width = mosaic2.shape[1] * 2
    delta_w = ((width - mosaic1.shape[1])//2) + 300
    delta_h = ((height - mosaic1.shape[0])//2) + 300
    color = [0, 0, 0]
    mosaic1 = cv2.copyMakeBorder(mosaic1, delta_h, delta_h, delta_w, delta_w, cv2.BORDER_CONSTANT,
                              value=color)
    mosaic2 = cv2.copyMakeBorder(mosaic2, delta_h, delta_h, delta_w, delta_w, cv2.BORDER_CONSTANT,
                              value=color)

    denoise1 = img1
    denoise2 = img2

    # Load the homography
    next_H = np.load('homography_submosaic_1.npy')
    prev_H = np.load('homography_submosaic_2.npy')

    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detectAndCompute(denoise1, None)
    kp2, des2 = surf.detectAndCompute(denoise2, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test and save the keypoints index
    i = 0
    good = []
    index_keypoints = []
    while len(good) <= MIN_MATCH_COUNT:
        print('Threshold:', threshold)
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append(m)
                index_keypoints.append(i)
            i = i + 1
        threshold = threshold + 0.1

    # cv.drawMatchesKnn expects list of lists as matches.
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3), plt.show()

    if len(good) >= MIN_MATCH_COUNT:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 20.0)
        H = np.dot(H, next_H)

        if H is None:
            print('Previous H')
            H = Hs[0]
            
        d = np.linalg.det(H)
        d2 = np.linalg.det(H[:1, :1])
        if d > 0.2:
            print('det OK!')
            dst = cv2.warpPerspective(mosaic2, linalg.inv(H), (mosaic2.shape[1], mosaic2.shape[0]))
            plt.imshow(dst), plt.show()
            final_img = overlap(mosaic1, dst)
            plt.imshow(final_img), plt.show()
            cv2.imwrite("merged.jpg", final_img)
        else:
            print('det < 0')

def overlap(img1, img2):
    result = np.where(np.all(img2 > 5, -1))
    img1[result] = img2[result]
    return img1

def crop_img(image):
    image_data = np.asarray(image)
    image_data_bw = image_data.min(axis=2)
    non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    image_data_new = image_data[cropBox[0]:cropBox[1] + 1, cropBox[2]:cropBox[3] + 1, :]
    crop = Image.fromarray(image_data_new)
    crop = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2RGB)
    return crop

# Go to the directory
directory = './submosaic/'
os.chdir(directory)

# Read the two submosaics
sub1 = cv2.imread("first_submosaic.jpg")
sub2 = cv2.imread("second_submosaic.jpg")

# Read the images where the submosaics are interrupted
images = np.load("frame_to_merge.npy", allow_pickle=True)
img1 = images[0,0,:]
img2 = images[0,1,:]
img3 = images[1,0,:]
img4 = images[1,1,:]

merge_submosaic(img3, img4, sub1, sub2)
