import nibabel as nib
import numpy as np
from sklearn.linear_model import Lasso,LassoCV
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

from orelmisc import n_max,n_test_max,testpre,trainpre,saveCSV,y_org
from preprocess import flatten,flatten_each_sample,loadData
d = 88*88*104 # 805376

""" ====================== Model and Prediction =========================== """

## TODO:
def fi_prediction(x,x_t,y):
  lasso = LassoCV(cv=3)
  y_t_pred = lasso.fit(x,y).predict(x_t)
  r = lasso.score(x_t,y_t_pred) # TODO: what is score???
  print("Tried alphas: %s"%lasso.cv_alphas_)
  return y_t_pred,r

"""
Main function. It does all the stuff.
"""
def doStuff(name,scale=1,P=100):
  y = y_org
  
  
  
  # Step 0: load data (zoomed or not)
  print("Reading training data.")
  x = loadData(scale=scale,train=True)
  
  print("Reading test data.")
  x_t = loadData(scale=scale,train=False)
  nTest = len(x_t)
  d = len(x[0,:])
  d2 = d/P





  # Step 1: divide data set into P subsets (TODO: boosting?)
  # we use P estimators
  sum_preds = np.zeros(nTest)
  sum_r = 0
  for i in range(0,P):
    if i %5==0:
      print("Make prediction for subset/estimator %s..."%(i+1))
    minI = d2*i
    maxI = min((d2)*(i+1),d)-1
    x_fi = x[:,i::P]
    x_t_fi = x_t[:,i::P]
    # Step 2: apply each estimator on data matrix x_fi and age vector y
    y_pred_i,r = fi_prediction(x_fi,x_t_fi,y)

    # insert prediction for test data i here
    sum_preds = sum_preds + y_pred_i
    sum_r = sum_r + r
    
  
  """ Step 3: combine all estimates """
  y_t_pred = [i / float(P) for i in sum_preds]
  r = sum_r / float(P)
    
  
  
  """Step 4: post-process and save prediction"""
  # TODO: calculate and sum of save covariances
  prefix = "%s_CV_P%s_zoom%sFULL"%(name,P,1/float(scale))
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
  print("Average of coefficients: %s"%r)
  # retuns a colleciton of stuff to return
  return (x,y,x_t,y_t_pred,y_t_pp,r)

result = doStuff(name="withCV",P=100,scale=1)
