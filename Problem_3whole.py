# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 12:30:09 2022

@author: pendl
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pandas as pd
import random

Base_path="C:/Users/pendl/Desktop"

data = pd.read_csv(Base_path+"/673_hw1/ENPM673_hw1_linear_regression_dataset - Sheet1.csv")
  
#converting column data to list
age = np.array(data['age'].tolist())
charges = np.array(data['charges'].tolist())
n=np.shape(age)[0]
ones = np.ones(n)
 
X = np.vstack((age, charges)).T
x = X[:,0]
y = X[:,1]
x_min = np.min(x)
x_max = np.max(x)

X_norm = (X - np.mean(X, axis = 0))
cov_mat = (1/np.shape(age)[0])*np.dot(X_norm.T, X_norm)
eigenvalues, eigenvectors =  LA.eig(cov_mat)

plt.scatter(X_norm[:,0],X_norm[:,1],marker='^',alpha=0.7,c='m')
origin=[0,0]
plt.quiver(*origin, *eigenvectors[0], color=['r'], scale=21)
plt.quiver(*origin, *eigenvectors[1], color=['b'], scale=21)
plt.show()

def plot_LS():
    
    
    o = np.ones(age.shape)
    z = np.vstack((x, o)).T

    D=np.matmul(np.linalg.inv(np.matmul(z.T, z)), np.matmul(z.T, y))  
    
    x_curve = np.linspace(x_min, x_max, 300) 
    o_curve = np.ones(x_curve.shape)
    z_curve = np.vstack(( x_curve, o_curve)).T
    y_curve = np.dot(z_curve, D)

    plt.scatter(x,y,marker='^',alpha=0.7,c='m')
    plt.plot( x_curve, y_curve, c='black',linestyle='-')
    plt.show()
    
    return 0
def plot_TLS():

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    o = np.ones(age.shape)
    u1 = x-x_mean
    u2 = y-y_mean
    
    U = np.vstack(((x - x_mean), (y - y_mean))).T
    A = np.dot(U.transpose(), U)
    
    B = np.dot(A.transpose(), A) 
    w, v = np.linalg.eig(B)
    index = np.argmin(w) 
    D = v[:, index]
        
    y_tls = []
    plt.scatter(X[:, 0], X[:, 1],marker='^',alpha=0.7,c='m')
    
    d = D[0]*x_mean+D[1]
    for i in range(0, len(x)):
        y_ = (d-(D[0]*x[i])) / D[1]
        y_tls.append(y_)

    plt.plot(x, y_tls,c='black',linestyle='-')
    plt.show()
def dist(p_x,p_y,x,y):
    p1=np.asarray((p_x[0][0],p_y[0][0]))
    p2=np.asarray((p_x[0][1],p_y[0][1]))
    p3=np.asarray((x,y))
    return np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)

def plot_RANSAC():
    threshold= 9
    n_best = 0
    n_iter=2000
    count=0
    max_inliers=0
    for i in range(0, n_iter):
        # Select subset of points from the dataset - 4
      P_x, P_y = np.array([[random.choice(x), random.choice(x)]]), np.array([[random.choice(y), random.choice(y)]])
      count=0
      
      if P_x[0][0] != P_x[0][1]:
          for i in range(len(x)):
              #P_x,P_y,x[i],y[i]
              dis=dist(P_x,P_y,x[i],y[i])
              if dis < threshold:
                  count+=1
                  
      if count > max_inliers:
            max_inliers=count
            n_best_x,n_best_y = P_x,P_y
        

    m=(n_best_y[0][1]-n_best_y[0][0]) / (n_best_x[0][1]-n_best_x[0][0]) 
    b= n_best_y[0][0] - m*n_best_x[0][0]
    
    plt.scatter(X[:, 0], X[:, 1],marker='^',alpha=0.7,c='m')
    y_rans = [m*i+b for i in x]
    plt.plot(x, y_rans,c='black',linestyle='-')
    plt.show()

plot_LS()
plot_TLS()    
plot_RANSAC()



