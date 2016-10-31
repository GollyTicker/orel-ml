import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import sys
from os.path import isfile

# Usage:
# $ python -i precompute_zoom.py
# >>> loadZoomed(scale=2,train=True)
# Loads the precomputed zoomed training data from ./train_{scale}.npy or
# procomputes and saves the the precomputed zoom and returns the precomputation.
# If train is False then instead prceses the test data.
# An example run is found at the bottom of this file.

from orelmisc import n_max,n_test_max,d_3d_org,shape_3d_org,trainpre,testpre

""" Flattens a numpy array to a 1-D vector"""
def flatten(data):
  return data.ravel()
  
# flatten [n,a,b,c] to [n,a*b*c]
def flatten_each_sample(x):
  return x.reshape(-1,x.shape[0]).transpose()
  
def remove_time_dimension(data):
  s = data.shape
  return data.reshape(s[0],s[1],s[2])

def zoomIn(data,scale):
  if scale==1:
    return remove_time_dimension(data)
  else:
    return zoom(remove_time_dimension(data),1.0/scale)

def loadData(scale=1,train=True):
  if scale != 1:
    return flatten_each_sample(loadZoomed(scale=scale,train=train))
  else: # scale = 1
    return flatten_each_sample(precompute_and_save(scale=scale,
              testData=not train,trainData=train,save=False))
  
# loads the data for scaling factor {scale}. If train is true,
# then training data is fetched. Otherwise its test data.
# If the data and scaling isn't precomputed yet, it will be done now.
def loadZoomed(scale=2,train=True):
  fname = ("train" if train else "test") + postfixstr(scale) + ".npy"
  if isfile(fname):
    print("Loading precomputed data.")
    return np.load(fname)
  else:
    print("No precomputation found.")
    return precompute_and_save(scale=scale,trainData=train,testData=not train)
  

def postfixstr(scale):
  return "_scale%s"%(scale)

def dim(data3d):
  return reduce(lambda x,y:x *y,data3d.shape,1)

def precompute_and_save(scale=2,testData=True,trainData=True,save=True):
  postfix = postfixstr(scale)
  new_shape = map(lambda x:x/scale,shape_3d_org[:-1])
  d = reduce(lambda a,x:a*x,new_shape,1)
  
  print("Precomputing zoom with scale=%s, d=%s." %(scale,d))
  x_last = 0
  
  ls = [(n_max,trainpre,"train")] if trainData else []
  ls = ls + ([(n_test_max,testpre,"test")] if testData else[])
  
  for n_i,pre,t_str in ls:
    print("Zooming and calculating for n_i=%s."%n_i)
    x = np.zeros([n_i]+new_shape)
    for j in range(n_i):
      i = j+1
      if j % 10==0:
        print("... processed %s of %s." %(j,n_i))
      filename = "%s%s.nii" % (pre,i)
      data = nib.load(filename).get_data()
      processed = zoomIn(data,scale)
      assert(dim(processed) == d)
      x[j,:,:,:] = processed
    x_last = x
    
    if save:
      sys.stdout.write("Saving X of dim=%s ..." % str(x.shape))
      fname = save_nd_array(x,t_str+postfix)
      print(" into %s done."%fname)
    
  return x_last
  
def save_nd_array(x,postfix):
  fm = postfix+".npy"
  np.save(fm,arr=x)
  return fm


"""
Precomputing training data the first time:

  >>> train = loadZoomed(scale=2,train=True)
  No precomputation found.
  Precomputing zoom with scale=2, d=805376.
  Zooming and calculating for n_i=278.
  ... processed 0 of 278.
  ... processed 10 of 278.
  ... processed 20 of 278.
  ... processed 30 of 278.
  ... processed 40 of 278.
  ... processed 50 of 278.
  ... processed 60 of 278.
  ... processed 70 of 278.
  ... processed 80 of 278.
  ... processed 90 of 278.
  ... processed 100 of 278.
  ... processed 110 of 278.
  ... processed 120 of 278.
  ... processed 130 of 278.
  ... processed 140 of 278.
  ... processed 150 of 278.
  ... processed 160 of 278.
  ... processed 170 of 278.
  ... processed 180 of 278.
  ... processed 190 of 278.
  ... processed 200 of 278.
  ... processed 210 of 278.
  ... processed 220 of 278.
  ... processed 230 of 278.
  ... processed 240 of 278.
  ... processed 250 of 278.
  ... processed 260 of 278.
  ... processed 270 of 278.
  Saving X of dim=(278, 88, 104, 88) ... into train_scale2.npy done.
  >>> train2 = loadZoomed(scale=2,train=True)   # every-call will now load instead
  Loading precomputed data.
  >>> np.array_equal(train,train2)
  True


Precomputing test data the first time:

  >>> test = loadZoomed(scale=2,train=False)
  No precomputation found.
  Precomputing zoom with scale=2, d=805376.
  Zooming and calculating for n_i=138.
  ... processed 0 of 138.
  ... processed 10 of 138.
  ... processed 20 of 138.
  ... processed 30 of 138.
  ... processed 40 of 138.
  ... processed 50 of 138.
  ... processed 60 of 138.
  ... processed 70 of 138.
  ... processed 80 of 138.
  ... processed 90 of 138.
  ... processed 100 of 138.
  ... processed 110 of 138.
  ... processed 120 of 138.
  ... processed 130 of 138.
  Saving X of dim=(138, 88, 104, 88) ... into test_scale2.npy done.
  >>> test2 = loadZoomed(scale=2,train=False)
  Loading precomputed data.
  >>> np.array_equal(test,test2)
  True

"""
