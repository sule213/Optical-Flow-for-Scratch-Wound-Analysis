#Individual results polar histogram
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import pandas as pd
from skimage import io, measure
from skimage.registration import optical_flow_tvl1, optical_flow_ilk
from numpy import arccos, array
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import thresholding
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sys import exit

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

# function to compute angle of a vector
def theta(v, w):
    # return arccos(v.dot(w)/(norm(v)*norm(w)))
    return (np.math.atan2(np.linalg.det([v, w]), np.dot(v, w)))

# function to compute magnitude of a vector
def magnitude(vector):
    return math.sqrt(sum(pow(element, 2) for element in vector))

# function to normalize angle 0 to 360 for negatives between -pi and pi
def normalize_angle_positive(angle):
    #if angle < 0:
        #return + math.ceil( -angle / 360 ) * 360
    return angle % 360

# function to segment the first frame to get the scratch direction
def segment(img):
    DS = 200  # disk size
    img = np.array(img, dtype=np.uint8)  # gray image
    img_entropy = entropy(img, disk(DS))
    # threshold the entropy image using Otsu to find threshold value
    img_thresh = thresholding.threshold_otsu(img_entropy)
    # create binary image with all pixels below the threshold
    return img_entropy <= img_thresh

# function to calculate the orientation of the scratch as base vector
def reg_prop(img):
    labels = measure.label(np.array(img, dtype=np.uint8))  # gray image
    df = pd.DataFrame(measure.regionprops_table(labels, img, properties=['orientation']))
    prop_theta = df._get_value(0, 'orientation')
    #return [math.cos(prop_theta), math.sin(prop_theta)]
    return [(math.sin(prop_theta)), -(math.cos(prop_theta))]

# last frame
fr = 60

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

    # list files
    im_files = [f for f in sorted(os.listdir(dir_in_name)) if f.endswith('.tif')]

   # read image
    for i in im_files:
      im1 = io.imread(os.path.join(dir_in_name, i))
      im1 = np.moveaxis(im1, 0, -1)  # [z,x,y] -> [x,y,z]
      # im = im[:,0:400,0:47]
      base = reg_prop((segment(im1[:, :, 0])))
      bases.append(base)
      print('base vector:', bases)   
      im = im1[:, :, 0:fr]  # 20, 40, 60, 80
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
          #imm = np.hypot(imfx, imfy)  # magnitude
          imfxt[:, :, j] = imfx
          imfyt[:, :, j] = imfy

        # mean magnitude
        imfxm = np.mean(imfxt, axis=2)
        imfym = np.mean(imfyt, axis=2)
        imfxs.append(imfxm)
        imfys.append(imfym)

        # plot
        nvec = 20  # number of vectors to be displayed along image dimension
        nl, nc = imfxm.shape
        step = max(nl//nvec, nc//nvec)
        y, x = np.mgrid[:nl:step, :nc:step]
        for i in range(0, len(imfxs)):
            imfxm = imfxs[i]
            imfym = imfys[i]
            u_ = imfxm[::step, ::step]
            v_ = imfym[::step, ::step]
            plt.figure(i+1)
            plt.imshow(np.mean(im, axis=2), cmap='gray')
            plt.title(names[i])
            plt.quiver(x, y, u_, v_, color='r', units='dots',
                    angles='xy', scale_units='xy', lw=10, width=2)
            plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_orien_1_q'+str(fr)+'.png')),
                        bbox_inches='tight')
            # plt.show()
            # plt.close()

            angles = []
            vc = np.zeros(2)
            for k in range(0, len(u_)):
                for l in range (0, len(v_)):
                    # array for measuring angle
                    arr = np.asarray([u_[l][k], v_[l][k]])
                    v = np.array([u_[l][k], v_[l][k]])                    
                    v = v/np.linalg.norm(v)
                    if np.isnan(v[0]):
                        v = np.zeros(2)
                    #print(v)
                    vc = vc + v                    
                    #compute magnitude of vector and remove zeros
                    if magnitude(arr) != 0:
                        # using base vector o [0,-1]
                        # bases = array([0,-1])
                        # angle between vectors
                        # convert angles from radians to degrees
                        angle = np.degrees(theta(bases[i], arr))
                        # angle = np.degrees(theta(bases, arr))
                        # normalize angles from -pi and pi to 0 and 360
                        angle = normalize_angle_positive(angle)
                        angles.append(angle)
                        # print(angle)
                        # print('------')
                # print("Angle values :", angles)
            # print(vc)
            print(np.degrees(theta(bases[i], vc)))  
            # exit(0)
            bin_size = 15
            # histogram normalized
            a, b = np.histogram(angles, bins=np.arange(0, 360+bin_size, bin_size), 
                                density=True)
            # number of occurrence of the angles in each bin
            centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])
            # plot polar histogram
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='polar')
            ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0,
                    align='center', color='.6', edgecolor='k')
            # set 0 on North of polar axis
            ax.set_theta_zero_location("N")
            # set angles anti-clockwise
            ax.set_theta_direction(-1)
            min_ylim = 0
            max_ylim = 0.016
            ax.set_ylim(min_ylim, max_ylim)
            plt.title(names[i].replace('.tif', ''))
            # plt.savefig(str(names[i])+"_polar_hist.png", bbox_inches='tight')
            plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_polar_hist_'+str(fr)+'.png')),bbox_inches='tight')
            # plt.show()
            plt.close()