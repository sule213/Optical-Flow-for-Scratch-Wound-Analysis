import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.registration import optical_flow_tvl1
from skimage.registration import optical_flow_ilk
from numpy import arccos, array
from numpy.linalg import norm

def normalize(x):
  return (x - x.min()) / (x.max() - x.min())

def theta(v,w):
  #return arccos(v.dot(w)/(norm(v)*norm(w)))
  return(np.math.atan2(np.linalg.det([v,w]),np.dot(v,w)))

if __name__ == '__main__':

  # parameters
  dir_in_name = '/home/campus.ncl.ac.uk/nsm368/Documents/Repo/3D_processing/scratch_wound/in/'
  dir_out_name = '/home/campus.ncl.ac.uk/nsm368/Documents/Repo/3D_processing/scratch_wound/out/'

  # variables
  names = []
  ims = []
  imfxs = []
  imfys = []

  # list files
  im_files = [f for f in os.listdir(dir_in_name) if f.endswith('.tif')]

  # read image
  for i in im_files:
    im = io.imread(os.path.join(dir_in_name, i))
    im = np.moveaxis(im, 0, -1)  # [z,x,y] -> [x,y,z]
    #im = im[:,0:400,0:47]
    im = im[:,:,0:9]  # 19, 39, 59, 79
    ims.append(im)
    names.append(i)

  for i in range(0, len(ims)):
    #i = 0
    print(names[i])
    # normalize
    im = normalize(ims[i])
    # compute the optical flow
    imfxt = np.zeros_like(im, shape=[im.shape[0],im.shape[1],im.shape[2]-1])
    imfyt = np.zeros_like(im, shape=[im.shape[0],im.shape[1],im.shape[2]-1])
    for j in range(im.shape[2]-1):
      print(str(im.shape[2]-1) + ': ' + str(j))
      #imfy, imfx = optical_flow_tvl1(im[:,:,j], im[:,:,j+1])
      imfy, imfx = optical_flow_ilk(im[:,:,j], im[:,:,j+1])
      imm = np.hypot(imfx, imfy) # magnitude
      imfxt[:,:,j] = imfx
      imfyt[:,:,j] = imfy

    # mean magnitude
    imfxm = np.mean(imfxt, axis=2)
    imfym = np.mean(imfyt, axis=2)
    imfxs.append(imfxm)
    imfys.append(imfym)

  # plot
  nvec = 8  # number of vectors to be displayed along each image dimension
  nl, nc = imfxm.shape
  step = max(nl//nvec, nc//nvec)
  y, x = np.mgrid[:nl:step, :nc:step]
  for i in range(0, len(imfxs)):
    imfxm = imfxs[i]
    imfym = imfys[i]
    u_ = imfxm[::step, ::step]
    v_ = imfym[::step, ::step]
    plt.figure(i+1)
    plt.imshow(np.mean(im, axis=2),cmap='gray' )
    plt.title(names[i])
    plt.quiver(x, y, u_, v_, color='r', units='dots', 
            angles='xy', scale_units='xy', lw=10, width=2)
    plt.savefig(os.path.join(dir_out_name, names[i].replace('.tif','_orien_1.png')),bbox_inches='tight')
  plt.show()

  print(v_)
  print(len(u_))
  radii = []
  angles = []
  for k in range(0, len(u_)):
    for l in range (0, len(v_)):
      #array for measuring angle
      arr = np.asarray([u_[k][l], v_[k][l]])
      # base vector [1,0]
      base = array([1,0])
      # angle between vectors
      angle = theta(base, arr)
      angles.append(angle)
      # Get the magnitude of the vector
      #radi = np.linalg.norm(arr)
    # remove rows having all zeroes
    ang = [a for a in angles if a != 0]
    #normalize radians to 1
    #rd = [i/sum(radii) for i in rd]
  #   plt.rcParams["figure.figsize"] = [7.00, 3.50]
  #   plt.rcParams["figure.autolayout"] = True
  #   plt.title(names[i])
  #   #width = np.pi / 16
  #   bin_size = 20
  #   bins = np.linspace(0.0, 2 * np.pi, bin_size + 1)
  #   width=np.deg2rad(bin_size)
  #   ax = plt.subplot(111, projection='polar')
  #   bars = ax.bar(ang, rd, width=width, bottom=0.0, color='red', edgecolor='k')
  #   #ax.set_ylim(0,1)
  # for r, bar in zip(rd, bars):
  #   #bar.set_facecolor(plt.cm.rainbow(r / 10.0))
  #   #bar.set_alpha(0.5)
  #   #bar.set_edgecolor('k')
  #   #plt.savefig(str(names[i])+"_circular_histogram_.png",bbox_inches='tight')
    #plt.show()
  bin_size = 20
  a , b = np.histogram(np.rad2deg(ang), bins=np.arange(0, 360+bin_size, bin_size), normed=True)
  centers = np.deg2rad(np.ediff1d(b)//2 + b[:-1])

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='polar')
  ax.bar(centers, a, width=np.deg2rad(bin_size), bottom=0.0, color='.8', edgecolor='k')
  ax.set_theta_zero_location("E")
  ax.set_theta_direction(1)
  ax.set_ylim(0,0.01)
plt.show()
