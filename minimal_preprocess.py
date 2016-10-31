import nibabel as nib
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom

from orelmisc import n_max,n_test_max,testpre,trainpre,saveCSV,y_org
from preprocess import flatten,flatten_each_sample,loadData

""" ====================== Model and Prediction =========================== """
reset = False
"""
Main function. It does all the stuff.
"""

prep = lambda i: int(i)
x_org = []
x_t_org = []
d_org = 0
y = y_org
y_t_pred = []
y_t_pp = []
alpha = 50 # TODO: guess size by looking at sizes of variables of transformation
prefix = ""
result = 0

def loadOrgData():
  global x_org,x_t_org,d_org
  if x_org == []:
    x_org = loadData(scale=1,train=True)
    x_t_org = loadData(scale=1,train=False)
    d_org = len(x_org[0,:])
    reset = False
  else:
    print "Data already loaded."


""" ======================== """
def preProcess():
  global d,x,x_t
  
  assert(x_org.shape==(n_max,d_org))

  d = 5 # keep more dimensions empty for later
  
  print "Preproessing... d = %s" % d
  x = np.full((n_max,d),0)
  x_t = np.full((n_max_test,d),0)
  
  x[:,0] = np.mean(x_org,axis=1)
  x_t[:,0] = np.mean(x_t_org,axis=1)
  
  x[:,1] = np.std(x_org,axis=1)
  x_t[:,1] = np.std(x_t_org,axis=1)
  
""" =================== """


def makePrediction():
  global y_t_pred,result
  print "Prediction with alpha = %s" % alpha
  prefix = "%s_alpha%s_FULL"%(name,alpha)
  lasso = Lasso(alpha=alpha)
  y_t_pred = lasso.fit(x,y).predict(x_t)
  r = lasso.score(x_t,y_t_pred)
  print("score r = %s"%r)
  print "Intercept: %s" % lasso.intercept_
  print "Coefficients: %s" % lasso.coef_

def doStuff(name,scale=1,P=100):
  loadOrgData()
  
  preProcess()
  
  makePrediction()
  
  y_t_pp = savePrediction()
  
  visualize(y,y_t_pp,prefix)
  
  print " ========= x ========== \n%s\n" % x
  print " ========= y_t_pp ========== \n%s\n" % y_t_pp
  
  # retuns a colleciton of stuff to return
  print "Variables available in 'result'"
  result = (x,y,x_t,y_t_pred,y_t_pp)

def visualize(y,y_t_pp,prefix):
  """ Step 5: make histogram plot of age """
  # (because no visualization for flat data matrix...)
  plt.hist(y,color="black",rwidth=0.7)
  #plt.hist(y_pred,color="darkgreen",rwidth=0.5)
  plt.hist(y_t_pp,color="darkblue",rwidth=0.5)
  plt.legend(["ages given for X",
    #"ages predicted for X",
    "ages predicted for X_t"])
  savedPlotFname = prefix + ".png"
  plt.savefig(savedPlotFname)
  print("Saved age diagram in %s"%savedPlotFname)
  plt.clf()

def savePrediction():
  global y_t_pp
  y_t_pp = [prep(i) for i in y_t_pred]
  savedFilename = saveCSV(y_t_pp,prefix)
  print("Saved predictions into %s" % savedFilename)
  return y_t_pp

# doStuff(name="noname")
