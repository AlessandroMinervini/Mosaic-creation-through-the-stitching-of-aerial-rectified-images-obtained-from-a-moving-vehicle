import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from math import *
from numpy import linalg
import imutils


#Setting directory
directory = './rectified/'

# Read the images rectified
def read_images():
    index = 1
    images = []
    for filename in sorted(os.listdir(directory)):
        if index < 0:
            print(index)
            index = index + 1
            continue
        i = cv2.imread(os.path.join(directory, filename))
        if i is not None:
            print(filename)
            height, width, _ = i.shape
            i = cv2.resize(i, (width // 1, height // 1))
            images.append(i)
    return images

# Search keypoints, compute homography and overlap the mosaic
def stitching(img1, img2, threshold, ind, Hs, skip, prev, result):
    MIN_MATCH_COUNT = 4
    skips = skip
    Hs = Hs
    end_skip = False
    end_prev = False
    end_sub = False
    subs = []
    size_submosaic = 33

    if (ind % size_submosaic) == 0:
        end_sub = True
        final_img = img2
        subs = img1
        H = Hs
        cv2.imwrite("submosaic/submosaic" + str(ind) + ".jpg", result)
        return final_img, H, skips, end_skip, subs, prev, end_prev, end_sub

    # Padding the images
    height= result.shape[0]*2
    width = result.shape[1]*2
    delta_w = ((width - img2.shape[1])//2) + 300
    delta_h = ((height - img2.shape[0])//2) + 300
    color = [0, 0, 0]
    img1 = cv2.copyMakeBorder(img1, delta_h, delta_h, delta_w, delta_w, cv2.BORDER_CONSTANT,
                              value=color)
    img2 = cv2.copyMakeBorder(img2, delta_h, delta_h, delta_w, delta_w, cv2.BORDER_CONSTANT,
                                value=color)
    result = cv2.copyMakeBorder(result, delta_h, delta_h, delta_w, delta_w, cv2.BORDER_CONSTANT,
                                value=color)

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray2 = clahe.apply(gray2)
    denoise1 = img1
    denoise2 = gray2

    # Initiate SURF detector
    surf = cv2.xfeatures2d.SIFT_create()

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
    while len(good)<= MIN_MATCH_COUNT:
        print('Threshold:',threshold)
        good = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good.append(m)
                index_keypoints.append(i)
            i = i+1
        threshold = threshold + 0.1

    # Function cv.drawMatchesKnn expects list of lists as matches.
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    # Draw the matched points
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    plt.imshow(img3), plt.show()

    if len(good) >= MIN_MATCH_COUNT:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, masked = cv2.findHomography(src, dst, cv2.RANSAC, 75.0)
        
        if H is None:
            prev = prev + 1
            if prev == 3:
                print('break, 3 previous H')
                end_prev = True
                cv2.imwrite("submosaic/submosaic" + str(ind) + ".jpg", result)
                prev = 0
                final_img = img2
                subs = result
                return final_img, H, skips, end_skip, subs, prev, end_prev, end_sub
            else:
                print('Previous H')
                H = Hs
        else:
            H = concat_H(Hs, H)
            print(H)
            prev = 0

        d = np.linalg.det(H)
        d2 = np.linalg.det(H[:1,:1])
        if d>0.2:# and d2>0:
            print('det OK!')
            dst = cv2.warpPerspective(img2, linalg.inv(H), (img2.shape[1], img2.shape[0]))
            plt.imshow(dst), plt.show()
            final_img = overlap(result, dst)
            skips = 0
        else:
            print('det < 0, skip image')
            skips = skips + 1
            H = Hs
            if skips == 4 :
                print('break, 4 skip images')
                cv2.imwrite("submosaic/submosaic" + str(ind) + ".jpg", result)
                skips = 0
                end_skip = True
                final_img = img2
                subs = result
            else:
                final_img = result
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        final_img = result
    plt.imshow(final_img), plt.show()
    return final_img, H, skips, end_skip, subs, prev, end_prev, end_sub

# Overlap the mosaic with the new warped image
def overlap(img1, img2):
    res = np.where(np.all(img2 > 5, -1))
    img1[res] = img2[res]
    return img1

# Crop the image
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

# Concatenate the Homography matrix
def concat_H(Hs, H):
    Hs = Hs
    H = H
    if len(Hs) > 0:
        Htot = Hs+H
        #Htot = np.dot(Hs, H)
        return Htot
    else:
        return H


images = read_images()
l = len(images)-1
result = images[0]
ind = 1
Hs = []
skips = 0
prev = 0
subH = []
to_merge_1 = 0
to_merge_2 = 0
sub_mosaic = []
merge_frames = []

for i in range(0, l, 1):
    print('Iteration ' + str(i) + ' ---------------------------------- :')
    stitch, H, skip, end_skip, subs, prev, end_prev, end_sub = stitching(images[i], images[i+1], threshold=0.4, ind = ind,
                                                                      Hs=Hs, skip = skips, prev = prev, result=result)
    stitch = crop_img(stitch)
    result = stitch
    ind = ind + 1
    skips = skip
    Hs = H
    if end_skip:
        np.save('submosaic/homography' + str(i) + '.npy', Hs)
        subH.append(Hs)
        sub_mosaic.append(subs)
        Hs = []
        to_merge_1 = images[i-4]
        to_merge_2 = images[i]
        merge_frames.append([to_merge_1, to_merge_2])
    if end_prev:
        np.save('submosaic/homography' + str(i) + '.npy', Hs)
        subH.append(Hs)
        sub_mosaic.append(subs)
        Hs = []
        to_merge_1 = images[i-3]
        to_merge_2 = images[i]
        merge_frames.append([to_merge_1, to_merge_2])
    if end_sub:
        np.save('submosaic/homography' + str(i) + '.npy', Hs)
        subH.append(Hs)
        sub_mosaic.append(subs)
        Hs = []
        to_merge_1 = images[i - 1]
        to_merge_2 = images[i]
        merge_frames.append([to_merge_1, to_merge_2])


stitch = crop_img(result)
np.save('submosaic/homography' + str(i) + '.npy', Hs)
np.save('submosaic/frame_to_merge', merge_frames)
cv2.imwrite("submosaic/final_submosaic.jpg", result)
sub_mosaic.append(result)
plt.imshow(result), plt.show()



