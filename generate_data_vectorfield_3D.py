#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pylab
import pickle
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.optimize import minimize
from mpl_toolkits import mplot3d
from scipy.spatial.distance import pdist,squareform
import sklearn


# In[2]:


matplotlib notebook


# In[3]:


matplotlib nbagg


# In[4]:


x_ger = np.linspace(-1,1, 20)


# In[5]:


x1,x2 = np.meshgrid(x_ger,x_ger)


# In[6]:


x1= x1.flatten()


# In[7]:


x2= x2.flatten()


# In[8]:


x3 = np.zeros(x1.shape)


# In[9]:


X =np.array( [x1.T, x2.T, x3.T]).T


# In[10]:


X.shape


# In[11]:


t = np.linspace(0,1,50)
ep = t[1]-t[0]


# In[12]:


def ode(X,ep):
    X1=[]
    X2=[]
    X3= []
    
    x1_pre = X[:,0]
    x2_pre = X[:,1]
    x3_pre = X[:,2]
    
    X1.append(x1_pre)
    X2.append(x2_pre)
    X3.append(x3_pre)
    
    for i in range(len(t)-1):
        x1 = x1_pre + 2*ep*x3_pre**3
        x2 = x2_pre 
        x3 = x3_pre - 2*ep*x1_pre**3
        
        X1.append(x1)
        X2.append(x2)
        X3.append(x3)
        
        x1_pre = x1
        x2_pre = x2
        x3_pre = x3
    
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    return X1,X2,X3
    


# In[13]:


def ode_backward(X_data,ep):
    X1=[]
    X2=[]
    X3= []
    
    x1_pre = X_data[:,0]
    x2_pre = X_data[:,1]
    x3_pre = X_data[:,2]
    
    X1.append(x1_pre)
    X2.append(x2_pre)
    X3.append(x3_pre)
    
    for i in range(len(t)-1):
        x1 = x1_pre - 2*ep*x3_pre**3
        x2 = x2_pre 
        
        x3 = x3_pre + 2*ep*x1_pre**3
        
        
        X1.append(x1)
        X2.append(x2)
        X3.append(x3)
        
        x1_pre = x1
        x2_pre = x2
        x3_pre = x3
    
    X1 = np.array(X1)
    X2 = np.array(X2)
    X3 = np.array(X3)
    return X1,X2,X3
    


# In[14]:


X1,X2,X3 = ode(X,ep)


# In[15]:


X1_0 = X1[:,0]
X2_0 = X2[:,0]
X3_0 = X3[:,0]


# In[16]:


X1_at_T = X1[-1,:]
X2_at_T = X2[-1,:]
X3_at_T = X3[-1,:]


# In[17]:


X_data = np.zeros((400,3))


# In[18]:


X_data[:,0] = X1_at_T
X_data[:,1] = X2_at_T
X_data[:,2] = X3_at_T


# In[ ]:





# In[19]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(X[:,0],X[:,1],X[:,2], c=X[:,0],cmap=plt.cm.jet,alpha = 0.1);


ax.scatter3D(X_data[:,0],X_data[:,1],X_data[:,2], c=X[:,0],cmap=plt.cm.jet);


# In[20]:


N = X_data.shape[0]
N


# In[21]:



import random
from sklearn.utils import shuffle
input_data =shuffle(X_data)
batch_data=[]
batch_size = 4
for ind in range(0, len(input_data), batch_size):
    
    a= np.random.dirichlet(alpha=[1,1,1,1], size=1)
    #a = np.array([[1]])
    input_batch = input_data[ind:ind + batch_size]
    
    batch_data.append((a@input_batch).reshape(3,))


# In[22]:


new_data = np.array(batch_data)


# In[23]:


new_data.shape


# In[25]:


"""import pickle
with open('generate_vector_s_convex_suffle.pickle','wb') as f:
    pickle.dump([new_data,new_data[:,0]], f)"""


# In[26]:


[x1_back,x2_back, x3_back] = ode_backward(X_data,ep)


# In[27]:


x1_back.shape


# In[28]:


2*X_data[0][0]**3


# In[29]:


x0_1 = x1_back[:,1].T
x0_2 = x2_back[:,1].T
x0_3 = x3_back[:,1].T


# In[30]:


x1_back_T = x1_back[-1,:]
x2_back_T = x2_back[-1,:]
x3_back_T = x3_back[-1,:]


# In[31]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter3D(x1_back_T,x2_back_T,x3_back_T, c=X[:,0],cmap=plt.cm.jet,alpha =1);
#ax.scatter3D(X[:,0],X[:,1],X[:,2], c=X[:,0],cmap=plt.cm.jet,alpha = 0.1);


ax.scatter3D(x0_1,x0_2,x0_3,cmap=plt.cm.jet,marker = '*');
ax.scatter3D(X_data[:,0],X_data[:,1],X_data[:,2], c=X[:,0],cmap=plt.cm.jet,alpha = 0.1);


# In[28]:


Q = np.array([[1,0,0],[0,1,0]])
Q.shape


# In[29]:


X_back_data = np.zeros((400,3))
X_back_data[:,0 ] =x1_back_T
X_back_data[:,1 ] =x2_back_T
X_back_data[:,2 ] =x3_back_T


# In[ ]:




