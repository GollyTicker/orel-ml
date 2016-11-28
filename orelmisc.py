
import csv
import numpy as np

n_max = 278
n_test_max = 138
testpre = "set_test/test_"
trainpre = "set_train/train_"
mriX = 176
mriY = 208
mriZ = 176

# number of dimensions in original 3D MRI data set
d_3d_org = 6443008
shape_3d_org = [176,208,176,1]

asbool = lambda x: str(int(x) is 1)

def saveCSV(y,filename):
  fname = filename + ".csv"
  f = open(fname,"wb")
  wr = csv.writer(f,delimiter=",")
  wr.writerow(["ID","Sample","Label","Predicted"])

  for i in range(len(y[:,0])):
    wr.writerows([ [i*3,i,"gender",asbool(y[i,0])],
                   [i*3+1,i,"age",asbool(y[i,1])],
                   [i*3+2,i,"health",asbool(y[i,2])] ])
  
  f.close()
  return fname
