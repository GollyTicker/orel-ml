
import csv
import numpy as np

n_max = 278
n_test_max = 138
testpre = "set_test/test_"
trainpre = "set_train/train_"

# number of dimensions in original 3D MRI data set
d_3d_org = 6443008
shape_3d_org = [176,208,176,1]

# read ages from targets.csv
y_org = np.array( [int(s[0]) for s in csv.reader(open("targets.csv"))] )

def saveCSV(xs,filename):
  fname = filename + ".csv"
  f = open(fname,"wb")
  wr = csv.writer(f,delimiter=",")
  wr.writerow(["ID","PREDICTION"])
  wr.writerows( [ [i+1,age] for i,age in enumerate(xs) ] )
  f.close()
  return fname
