import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.registration import optical_flow_tvl1
from skimage.registration import optical_flow_ilk
from skimage import exposure


def normalize(x):
  return (x - x.min()) / (x.max() - x.min())

if __name__ == '__main__':

  # parameters
  dir_in_name = '/home/campus.ncl.ac.uk/nsm368/Documents/Repo/3D_processing/scratch_wound/in/'
  dir_out_name = '/home/campus.ncl.ac.uk/nsm368/Documents/Repo/3D_processing/scratch_wound/out/'

  # variables
  names = []
  ims = []
  imms = []

  # list files
  im_files = [f for f in os.listdir(dir_in_name) if f.endswith('.tif')]

  # read image
  for i in im_files:
    im = io.imread(os.path.join(dir_in_name, i))
    im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]
    #im = im[0:50,0:50,20:25]
    im = im[:,:,0:9]  # 19, 39, 59, 79

    ims.append(im)
    names.append(i)

  for i in range(0, len(ims)):
    #i=0
    print(names[i])
    # normalize
    im = normalize(ims[i])
    # compute the optical flow
    immt = np.zeros_like(im, shape=[im.shape[0],im.shape[1],im.shape[2]-1])
    for j in range(im.shape[2]-1):
      print(str(im.shape[2]-1) + ': ' + str(j))
      #imfy, imfx = optical_flow_tvl1(im[:,:,j], im[:,:,j+1])
      imfy, imfx = optical_flow_ilk(im[:,:,j], im[:,:,j+1])
      imm = np.hypot(imfx, imfy) # magnitude
      immt[:,:,j] = imm

    # mean magnitude
    imm = np.mean(immt, axis=2)
    imms.append(imm)

    # find motion range
    imm = np.asarray(imms)
    #r1 = imm.min()
    #r2 = imm.max()
    r1 = 0
    r2 = 75

    for i in range(0, len(imms)):
      plt.figure(i+1)
      plt.imshow(imms[i], cmap='jet', vmin=r1, vmax=r2)
      plt.title(names[i])
      plt.colorbar()
    plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_mot_q4.png')),bbox_inches='tight')
    #plt.show()