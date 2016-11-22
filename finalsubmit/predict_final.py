
# coding: utf-8

# # Random Forest (score <= 0.31)
# ## MLP2 - Team 0rel
# ### Alessandra Urbinati, Cristina Bozzo, Swaneet Sahoo

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi,sin,cos
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
import csv


""" =============== Constants and global variables ================== """
print "[0] Init"

n_max = 278
n_test_max = 138
testpre = "data/set_test/test_"
trainpre = "data/set_train/train_"

# target values form target.csv
y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# mean prediction (ca.  0.75 for all patients))
y_mean = np.full(n_max,np.mean(y),dtype=np.float64)[:n_test_max]

# dummy variables
y_t_pred = None
yts_pred = None
y_t_pp = None
prefix=None
result = None
xa = []
x = None
x_t = None

# for spherical coordinate system transformation
TO_RADIANS = pi/180
r_division = 6    # number of division for radius
theta_division = 4  # ... for theta (angle from vertical Z-axis to vector)
phi_division = 6  # ... for phi (angle in X-Y-plane)
# radius form 0 to 80
# theta from 0 to 180
# phi from 0 to 360
rMax = 80
tMin = 30
tMax = 180-tMin
ranges = [1,r_division,theta_division,phi_division]
nHists = r_division*theta_division*phi_division

# for histograms
space = 70  # 1 + size of histograms
hSize = space-1
bins = np.linspace(1,1700,space)  # the buckets are evenly spaced from 1 to 1700 (higher than observed maximum value)
ds = nHists*hSize

# names for output/input files
name = "src/"+str(space)+"_split_validated"
finalSubOut = "final_sub"
fname = "src/many_hists"+str(nHists)+"_space" + str(space) + ("_divs_%s_%s_%s.npy" % (r_division,theta_division,phi_division))
fnameSpherical = "src/spherical_every2.npy"
# xSpherical = np.load(fnameSpherical) #  not needed for actual prediction anymore. was part of preprocessing

print "  Number of Histograms:",nHists
print "  Size of Histograms:",hSize
print "  Dimensions: ",nHists*hSize


""" =========== Misc. functions =============== """

def saveCSV(xs,filename):
  fname = filename + ".csv"
  f = open(fname,"wb")
  wr = csv.writer(f,delimiter=",",lineterminator="\n")
  wr.writerow(["ID","Prediction"])
  wr.writerows( [ [i+1,pred] for i,pred in enumerate(xs) ] )
  f.close()
  return fname

prep_ = lambda i: cap(i) # post-processing
def prep(a):
  return map(prep_,a)

cap = lambda p: min(1,max(0,p)) # cap values into [0,1]


""" ================== PREPROCESSING ====================== """

def spherical2cart(r,theta,phi):
  return (r*sin(theta)*cos(phi),r*sin(theta)*sin(phi),r*cos(theta))

fromto = lambda di,sph: zip(np.round(np.linspace(0,sph.shape[di],ranges[di]+1)),np.round(np.linspace(0,sph.shape[di],ranges[di]+1))[1:])

# the code commented out in "loadAndPreprocess()" was used to transform the images into spherical coordinates
# We didn't supply the .npy file for the spherical coordinates, since it takes ~1h to compute and is a few GBs large.
# since we don't need it for the prediction itself anymore, we just provide the features data file. (many_hists144_space70_divs_6_4_6.npy)
def loadAndPreprocess():
    global xSpherical
    
    xa = np.zeros((n_max+n_test_max,ds))
    #xSpherical = np.zeros((n_max+n_test_max,rMax/2+1,(tMax-tMin)/2+1,360/2+1))
    
    i = 0
    
    while i < n_max+n_test_max:
      if i % 6 == 0:
        print "  i = %s ... %.1f%%" % (i,float(i)/(n_max+n_test_max)*100)
      
      n_i,pre,t_str = (n_max,"set_train/","train") if i < n_max else (n_test_max,"set_test/","test")
      filename = "%s%s_%s.nii" % (pre,t_str,i%n_max+1)
      Xtotal,Ytotal,Ztotal = (176,208,176)
      data = nib.load(filename).get_data().reshape((Xtotal,Ytotal,Ztotal))
      
      #print "===== Calculate spherical coordiantes ===="
      """Calculate zoomed spherical representation, needs 3 seconds
      for r in np.linspace(0,rMax,rMax/2+1):
        for theta in np.linspace(tMin,tMax,(tMax-tMin)/2+1):
          for phi in np.linspace(0,360,360/2+1):
            x,y,z = spherical2cart(r,theta*TO_RADIANS,phi*TO_RADIANS)
            x = x + Xtotal/2
            y = y + Ytotal/2
            z = z + Ztotal/2
            if 0 <= x < Xtotal and 0 <= y < Ytotal and 0 <= z < Ztotal:
              xSpherical[i,r/2,(theta-tMin)/2,phi/2] = data[int(x),int(y),int(z)]"""

      # calculate histograms: 8*4*8 = 256 histograms
      hCount = 0
      for l,u in fromto(1,xSpherical):
        l0,u0=(int(l),int(u))
        for l,u in fromto(2,xSpherical):
          l1,u1=(int(l),int(u))
          for l,u in fromto(3,xSpherical):
            l2,u2=(int(l),int(u))
            cut = xSpherical[i,l0:u0,l1:u1,l2:u2]
            h=np.histogram(cut.ravel(),bins=bins)[0]
            xa[i,(hCount*hSize):((hCount+1)*hSize)] = h
            hCount = hCount + 1
      i = i+1
    
    np.save(fname,xa)
    print "======= Saved data matrix xa into %s =========" % fname
    
    #np.save(fnameSpherical,xSpherical)
    #print "======= Saved spherical coordinates into %s =========" % fnameSpherical

# no need since matrix is already precomputed
# loadAndPreprocess()

xa = np.load(fname)
x,x_t = (xa[0:n_max,:],xa[n_max:,:])

print "[1] Preprocessing, ok"
print "[2] Feature selection, ok"

""" ======================== SPLIT TRAINING AND VALIDATION ================== """
print "[3] Splitting data into training and validation set"
# xa. all data
# x. public training data
# y. public training targets
# x_t public to be predicted data

# xtr. training data
# ytr. training targets
# xts. validation data
# yts. validation targets

# indices for splitting
ones = np.array(filter(lambda i: y[i]==1,range(0,n_max)))
zeros = np.array(filter(lambda i: y[i]==0,range(0,n_max)))
# split the indices
onestr,onests,_,_ = train_test_split(ones,ones*0,test_size=0.3,random_state=1)
zerostr,zerosts,_,_ = train_test_split(zeros,zeros*0,test_size=0.3,random_state=1)

# plot the data. a few random states have been tried to to make sure all splits are balanced enough.
print " \n Indices for total data: "
print "    healthy",len(ones),": ",ones[5:15],"..."
print "    sick",len(zeros),": ",zeros[5:15],"...\n"

print "  Indices for splitted data: "
print "    healthy training ",len(onestr),": ",onestr[5:15],"..."
print "    healthy test ",len(onests),": ",onests[5:15],"..."
print "    sick training ",len(zerostr),": ",zerostr[5:15],"..."
print "    sick test ",len(zerosts),": ",zerosts[5:15],"..."

xtr = np.vstack((x[onestr],x[zerostr]))
xts = np.vstack((x[onests],x[zerosts]))
ytr = np.hstack((y[onestr],y[zerostr]))
yts = np.hstack((y[onests],y[zerosts]))

print "\n  Splitted data into test and validation data\n"

""" ===================== MODEL ============================= """

def randomForest(n_est,f,msp,max_depth):
  global y_t_pred,yts_pred,result,y_t_pp
  prefix = "%s_RandomForest_n%s_feats%s_msp%s_max_depth%s"%(name,n_est,f,msp,max_depth)
  print "[4] =========== Prediction =========\n  with %s\n  This will take a few seconds..." % prefix
  model = RandomForestRegressor(n_est,max_features=f,min_samples_split=msp,max_depth=max_depth,random_state=1)
  xtr1 = xtr[:,:] # use all data
  xts1 = xts[:,:] # use all data
  x_t1 = x_t[:,:]
  yts_pred = model.fit(xtr1,ytr).predict(xts1)
  y_t_pred = model.predict(x_t1)
  
  y_t_pp = prep(y_t_pred)
  yts_pp = prep(yts_pred)
  ytr_pp = prep(model.predict(xtr1))
  ltr = log_loss(ytr,ytr_pp)
  lts = log_loss(yts,yts_pp)
  ltm = log_loss(yts,y_mean[:len(yts)])
  print("\n  ===============    Log-Loss train           ltr = %.3f   =========="%ltr)
  print(  "  ===============    Log-Loss validation      lts = %.3f   =========="%lts)
  print(  "  ===============    Log-Loss mean prediciton ltm = %.3f   ==========\n"%ltm)
  print "  Model Score: %s" % model.score(xts1,yts)
  prefix = "%s_expected_score%.3f"%(prefix,lts)
  return prefix,model


""" ============================== PREDICTION ============================= """

def doStuff(n_est=400,f=0.15,msp=10,max_depth=3):
  global prefix,result
  
  prefix,model = randomForest(n_est=n_est,f=f,msp=msp,max_depth=max_depth)
  
  savePrediction()
  
  visualize(y,y_t_pp,prefix)
  
  print "  Variables available in 'result'"
  result = (x,y,x_t,y_t_pred,y_t_pp,model)

def visualize(y,y_t_pp,prefix):
  plt.clf()
  plt.plot(np.array(y_t_pp),"bo")
  plt.savefig(prefix + "_predictions.png")
  plt.clf()
  plt.title("PREDICTIONS sorted (0=unhealty, 1=healthy)")
  plt.plot(sorted(y_t_pp),"bo")
  plt.savefig(prefix + "_plot.png")
  print("  Saved age diagrams as %s"%(prefix+"*.png"))

def savePrediction():
    saveCSV(map(cap,y_t_pp),prefix)
    savedFilename = saveCSV(map(cap,y_t_pp),finalSubOut)
    print("  Saved predictions into %s" % savedFilename)


# Make Prediction. These values were chosen by minimizing validation error (second Log-Loss statement in output)
# n_est: number of decision trees
# f: how much each of the trees gets to see (0.2 means 20%)
# msp: minimal number of samples required to split a node in the decision tree
# max_depth: maximum depth of the trees. after this no split are done.
doStuff(n_est=400,f=0.2,msp=10,max_depth=2)

print "[5] Finished."
