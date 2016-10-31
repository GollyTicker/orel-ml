import nibabel as nib
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

from orelmisc import n_max,n_test_max,testpre,trainpre,saveCSV,y_org
from preprocess import flatten,flatten_each_sample,loadData
from multiprocessing import Process
import sharedmem
d = 88*88*104 # 805376
d2 = 0
y=0
"""
from multiprocessing import Process
import numpy

def do_work(data, start):
    data[start] = 0;

def split_work(num):
    n = 20
    width  = n/num

    processes = [Process(target=do_work, args=(shared, i*width)) for i in xrange(num)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
        """

""" ====================== Model and Prediction =========================== """

## TODO:
def fi_prediction(x,x_t,y,alpha):
  lasso = Lasso(alpha=alpha)
  y_t_pred = lasso.fit(x,y).predict(x_t)
  r = lasso.score(x_t,y_t_pred) # TODO: what is score???
  return y_t_pred,r

def each_elem(i,x,x_t,results):
    if i%5==0:
      print("Make prediction for subset/estimator %s..."%(i+1))
    minI = d2*i
    maxI = min((d2)*(i+1),d)-1
    x_fi = x[:,minI:maxI]
    x_t_fi = x_t[:,minI:maxI]
    # Step 2: apply each estimator on data matrix x_fi and age vector y
    y_t_pred,r = fi_prediction(x_fi,x_t_fi,y,alpha=alpha)
    results[i] = (y_t_pred,r)



"""
Main function. It does all the stuff.
"""
def doStuff(name,alpha = 77,scale=1,P=100):
  y = y_org
  
  
  
  
  # Step 0: load data (zoomed or not)
  print("Reading training data.")
  #x = sharedmem.empty(n_max)
  x = loadData(scale=scale,train=True)
  
  print("Reading test data.")
  #x = sharedmem.empty(n_test_max)
  x_t = loadData(scale=scale,train=False)
  nTest = len(x_t)
  print x
  d = len(x[0,:])
  d2 = d/P





  """ Step 1: divide data set into P subsets (TODO: boosting?) """
  # we use P estimators
#  results = pool.map(each_elem,range(0,P))
  results = range(0,P)
  processes = [Process(target=each_elem, args=(i,x,x_t,results))
                  for i in range(0,P)]
  for p in processes:
      p.start()
  for p in processes:
      p.join()
  
  
  """ Step 3: combine all estimates """
  y_t_pred = reduce(lambda a,x:a+x[0],results,0) / float(P)
  r = reduce(lambda a,x:a+x[1],results,0) / float(P)
  
  
  """Step 4: post-process and save prediction"""
  # TODO: calculate and sum of save covariances
  prefix = "%s_alpha%s_P%s_zoom%sFULL"%(name,alpha,P,1/float(scale))
  prep = lambda i: int(i)
  y_t_pp = [prep(i) for i in y_t_pred]
  savedFilename = saveCSV(y_t_pp,prefix)
  print("Saved predictions into %s" % savedFilename)
  
  
  
  """ Step 5: make histogram plot of age """
  # (because no visualization for flat data matrix...)
  plt.hist(y,color="black",rwidth=0.7)
  #plt.hist(y_pred,color="darkgreen",rwidth=0.5)
  plt.hist(y_t_pp,color="darkblue",rwidth=0.5)
  plt.legend(["ages given for X","ages predicted for X","ages predicted for X_t"])
  savedPlotFname = prefix + ".png"
  plt.savefig(savedPlotFname)
  print("Saved age diagram in %s"%savedPlotFname)
  plt.clf()
  
  # retuns a colleciton of stuff to return
  return (x,y,x_t,y_t_pred,y_t_pp)

# making alpha dependent on the number of dimensions
# result = doStuff(name='lasso_snt',calculateAlpha=True,denom=800)

