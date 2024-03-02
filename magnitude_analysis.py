#Combined magnitude results
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import cv2
import sys
from skimage import io, measure, img_as_ubyte, img_as_float
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from numpy import arccos, array
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import thresholding
from skimage.filters.rank import entropy
from skimage.morphology import disk

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# Normalize between 0 and 1
def nrmlz(data):
    #return (data - np.min(data)) / (np.max(data) - np.min(data))
    return (data - 0) / (50)

# function to compute angle of a vector
def theta(v, w):
    # return arccos(v.dot(w)/(norm(v)*norm(w)))
    return (np.math.atan2(np.linalg.det([v, w]), np.dot(v, w)))

# function to compute magnitude of a vector
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

# function to normalize angle 0 to 360 for negatives between -pi and pi
def normalize_angle_positive(angle):
    return angle % 360

# last frame
fr = 80

if __name__ == '__main__':

    # parameters
    folder = sys.argv[1]
    dir_in_name = '/home/campus.ncl.ac.uk/nsm368/Documents/Repo/3D_processing/scratch_wound/in/'
    dir_out_name = '/home/campus.ncl.ac.uk/nsm368/Documents/Repo/3D_processing/scratch_wound/out/'
    dir_in_name = dir_in_name + '/' + folder + '/'
    dir_out_name = dir_out_name + '/' + folder + '/'
    print(dir_in_name)
    print(dir_out_name)

    # variables
    names = []
    ims = []
    imfxs = []
    imfys = []
    bases = []
    mags = []

    # list files
    im_files = [f for f in sorted(os.listdir(dir_in_name)) if f.endswith('.tif')]

    # read image
    for i in im_files:
      im = io.imread(os.path.join(dir_in_name, i))
      im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]

      im = im[:, :, 0:fr]
      ims.append(im)
      names.append(i)

    for i in range(0, len(ims)):
        # i = 0
        print(names[i])
        # normalize
        im = normalize(ims[i])
        # compute the optical flow
        imfxt = np.zeros_like(im, shape=[im.shape[0],im.shape[1],im.shape[2]-1])
        imfyt = np.zeros_like(im, shape=[im.shape[0],im.shape[1],im.shape[2]-1])
        for j in range(im.shape[2]-1):
          print(str(im.shape[2]-1) + ': ' + str(j))
          #imfy, imfx = optical_flow_tvl1(im[:,:,j], im[:,:,j+1])
          imfy, imfx = optical_flow_ilk(im[:, :, j], im[:, :, j+1])
          imm = np.hypot(imfx, imfy)  # magnitude
          imfxt[:, :, j] = imfx
          imfyt[:, :, j] = imfy

        # mean magnitude
        imfxm = np.mean(imfxt, axis=2)
        imfym = np.mean(imfyt, axis=2)
        imfxs.append(imfxm)
        imfys.append(imfym)

        # plot
        nvec = 20  # number of vectors to be displayed along each image dimension
        nl, nc = imfxm.shape
        step = max(nl//nvec, nc//nvec)
        y, x = np.mgrid[:nl:step, :nc:step]
        for i in range(0, len(imfxs)):
            imfxm = imfxs[i]
            imfym = imfys[i]
            u_ = imfxm[::step, ::step]
            v_ = imfym[::step, ::step]
            # plt.figure(i+1)
            # plt.imshow(np.mean(im, axis=2), cmap='gray')
            # plt.title(names[i])
            # plt.quiver(x, y, u_, v_, color='r', units='dots',
            #         angles='xy', scale_units='xy', lw=10, width=2)
            # plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_orien_1_q'+str(fr)+'.png')),
            #             bbox_inches='tight')
            #plt.show()

            mag = np.zeros(len(u_))
            for k in range(0, len(u_)):
                for l in range (0, len(v_)):
                #array for measuring angle
                    arr = np.asarray([u_[l][k], v_[l][k]])
                #compute magnitude of vector and remove zeros
                    if magnitude(arr) != 0:
                         if magnitude(arr) > 5:
                            if magnitude(arr) < 51:
                                mag = magnitude(arr)
                        #print(angle)
                        # normalize angles from -pi and pi to 0 and 360
                        #mag = normalize_angle_positive(angle)
                        #print(ang)
                                mags.append(mag)
                #print("Angle values :", mags)
                np.savetxt(str(folder)+"_"+str(fr)+"_magnitude_zoom.csv", mags, delimiter=",")

    bin_size = 15
    #mags = nrmlz(mags)
    # histogram normalized
    plt.ylabel('Frequency')
    plt.xlabel('Magnitude')
    #plt.title(names[i].replace('.tif', ''))
    plt.ylim([0, 0.1])
    #plt.xlim([0, 1])
    plt.hist(mags, bins=bin_size, density=True)
    #plt.hist(lst, nrmlz(mags), bins=bin_size, density=True)
    #     # plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_polar_hist_q'+str(fr)+'.png')),bbox_inches='tight')
    plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_polar_hist_mag_zoom'+str(fr)+'.png')),bbox_inches='tight')
    #plt.show()