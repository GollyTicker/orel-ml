import nibabel as nib
import numpy as np
import csv
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
  
"""
Usage: (you need to have numpy,scipy,scikit and matplotlib installed)
$ python
>>> execfile("pipeline.py")
>>> result = doStuff(15,15,alpha=0.3)
>>> result.x # shows data matrix
>>> result.y # shows target_values
>> result = doStuff(alpha=0.5) # use entire data set

A diagram is saved into *.png and the submittable predictions are in *.csv.
If a file is called try_0.1_FULL.csv then it means
that it used the entire data set and alpha=0.1 to calculate the predictions and the diagram.
The diagram shows a histogram of the ages.

Result object:
x_t => test data matrix
lasso => lasso object
y_t_pred => lasso prediction of the age for test data x
y_t_pp => processed prediction (e.g. age capping)

"""

n_max = 278
n_test_max = 138
testpre = "set_test/test_"
trainpre = "set_train/train_"
d = 6443008
r1 = [65,50,80]
r2 = [115,110,121]
# calculate dimension from 3D cuboid
#d = reduce(lambda x,a:x*a,[y-i for i,y in zip(r1,r2)],1)

# read ages from targets.csv
y_org = np.array( [int(s[0]) for s in csv.reader(open("targets.csv"))] )


""" ============== Preprocessing steps ========================= """

""" Flattens a numpy array to a 1-D vector"""
def flatten(data):
  return data.ravel()
  
def remove_time_dimension(data):
  s = data.shape
  return data.reshape(s[0],s[1],s[2])

""" Shrink 3D data into a small cuboid """
def shrink3D(data,range0,range1):
  return data[
          range0[0]:range1[0],
          range0[1]:range1[1],
          range0[2]:range1[2],
          ]

def dimension(data):
  s = data.shape
  prod = 1
  for i in s:
    prod = prod * i
  return prod

def process(data):
  return flatten(remove_time_dimension(data))

""" ====================== Model and Prediction =========================== """

def lassoRegression(alpha,x,y,x_t):
  lasso = Lasso(alpha=alpha)
  print("Running Lasso with alpha = %s ..." % alpha)
  y_t_pred = lasso.fit(x,y).predict(x_t)
  y_pred = lasso.predict(x)
  r2_score = lasso.score(x_t,y_t_pred)
  print("Lasso: %s"% lasso)
  print("coefficients: %s" % lasso.coef_)
  print("intercept: %s" % lasso.intercept_)
  print("score on test data: %s" % r2_score)
  return lasso,y_t_pred,y_pred

"""
Main function. It does all the stuff.
"""
def doStuff(name,kernel="rbf",n=n_max,n_test=n_test_max,alpha = 0.1):
  full = "_FULL" if n==n_max else ""
  prefix = "%s_%s%s"%(name,alpha,full)
  kpca = KernelPCA(kernel=kernel, fit_inverse_transform=True)
  
  """ Read training data """
  x_pre = np.zeros((n,d))
  y = y_org[:n]
  for j in range(n):
    i = j+1
    filename = "%s%s.nii" % (trainpre,i)
    data = loadArray(filename)
    processed = process(data)
    assert(len(processed) == d)
    x_pre[j,:] = processed
  
  print("Calculating PCA for training data...")
  x = kpca.fit_transform(x_pre)
  d2 = len(x_pre[0,:])
  print("Created data matrix X of size %sx%s" % (n,d2))
  
  """ Read test data """
  x_t_pre = np.zeros((n_test,d))
  for j in range(n_test):
    i = j+1
    filename = "%s%s.nii" % (testpre,i)
    data = loadArray(filename)
    processed = process(data)
    x_t_pre[j,:] = processed
  
  print("Calculating PCA for test data...")
  x_t = kpca.fit_transform(x_t_pre)
  print("Created test-data matrix X_t of size %sx%s" % (n_test,d2))
  # make prediction
  print("x: %s" % x)
  print("x_t: %s" % x_t)
  print("y: %s" % y)
  lasso,y_t_pred,y_pred = lassoRegression(alpha,x,y,x_t)
  
  """post-process and save prediction"""
  # bound entries by 18 and 105 and round to nearest integer
  # save as csv
  prep = lambda i: int(i) #max(min(int(round(i)),105),18)
  
  y_t_pp = [prep(i) for i in y_t_pred]
  savedFilename = saveCSV(y_t_pp,prefix)
  print("Saved predictions into %s" % savedFilename)
  
  """ make histogram plot of age """
  # (because no visualization for flat data matrix...)
  plt.hist(y,color="black",rwidth=0.7)
  plt.hist(y_pred,color="darkgreen",rwidth=0.5)
  plt.hist(y_t_pp,color="darkblue",rwidth=0.5)
  plt.legend(["ages given for X","ages predicted for X","ages predicted for X_t"])
  #plt.show(block=False)
  savedPlotFname = prefix + ".png"
  plt.savefig(savedPlotFname)
  print("Saved age diagram in %s"%savedPlotFname)
  plt.clf()
  
  # retuns a colleciton of stuff to return
  return Result(x,y,x_t,lasso,y_t_pred,y_t_pp,y_pred)

class Result:
  def __init__(self,x,y,xt,ls,y_t_pred,y_t_pp,y_pred):
    self.x=x
    self.y=y
    self.x_t=xt
    self.lasso=ls
    self.y_t_pred=y_t_pred
    self.y_t_pp=y_t_pp
    self.y_pred = y_pred

"""  ============ Helper functions ======================== """
def loadArray(filename):
  if filename[-3:] == "nii":
    return nib.load(filename).get_data()
  elif filename[-2:] == "1d":
    return np.loadtxt(open(filename,"rb"),delimiter=",")

def saveArray(data,filename):
  np.savetxt(filename,data,delimiter=",")

def saveCSV(xs,filename):
  fname = filename + ".csv"
  f = open(fname,"wb")
  wr = csv.writer(f,delimiter=",")
  wr.writerow(["ID","PREDICTION"])
  wr.writerows( [ [i+1,age] for i,age in enumerate(xs) ] )
  f.close()
  return fname
