import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.widgets import Slider, Button, RadioButtons
from preprocess import remove_time_dimension
# HOW TO USE:
#1. open command time/terminal in this folder
# run $ python -i show-interactive.py
# in pyhton interpreter you can view any image:
# >>> show_data("set_train/traint_1.nii",i)
# use i = 0 for left-to-right sliding
# use i = 1 for front-to-back sliding
# use i = 2 for top-to-bottom sliding
# use the correct path
# the smaller the windows the faster the image loads

# Have fun :)

a = 0
data = 0
shape = 0
i = 0
fig = 0
axes = 0
axsliderD = 0
ssliceD = 0
sliceD = 0

def show_data(filename,i_,custom=None):
  global a, data, shape, i, ssliceD, fig, axes, axsliderD
  i = i_
  if custom != None:
    data = custom
  else:
    a = nib.load(filename)
    data = remove_time_dimension(a.get_data())
  shape = data.shape
  print(shape)
  fig, axes = plt.subplots(1, 1)
  axsliderD = plt.axes([0.05, 0.05, 0.6, 0.03], axisbg='lightgoldenrodyellow')
  ssliceD = Slider(axsliderD, 'slice', 0, shape[i], valinit=shape[i]/2)
  ssliceD.on_changed(update)
  plt.show()


def update(val):
  global sliceD
  sliceD = ssliceD.val
  dt = 0
  if i==0:
    dt = data[sliceD,:,:]
  elif i==1:
    dt = data[:,sliceD,:]
  else:
    dt = data[:,:,sliceD]
  axes.imshow(dt.T, cmap="gray", origin="lower")
  fig.canvas.draw_idle()
#...



