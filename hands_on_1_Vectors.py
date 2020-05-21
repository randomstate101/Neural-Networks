#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.datasets
import matplotlib.colors 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


plt.quiver(0,0,3,4 , color='r')


# In[3]:


def plot_vectors(vecs):
    
    colors = [ 'r', 'b' , 'g' , 'y']
    i = 0 
    for vec in vecs:
        plt.quiver(vec[0], vec[1], vec[2], vec[3], scale_units ='xy', angles = 'xy' , scale =1, color = colors[ i % len (colors) ])
        i+= 1
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    


# In[4]:


plot_vectors([(0,0,4,3),(0,0,3,4),(0,0,5,5),(0,0,0,5)])


# In[5]:


vecs  = [np.array([0,0,4,4]),np.array([0,0,-6,-3]),np.array([0,0,-5,4]),np.array([0,0,6,-4])] 
                                                              
                                                                                   


# In[6]:


plot_vectors([vecs[0], vecs[1],vecs])


# In[7]:


a = np.array([5,6])
b = np.array([7,-12])


# In[8]:


a_b = np.dot(a,b)/np.linalg.norm(b)


# In[9]:


print (a_b)


# In[10]:


vec_c = (a_b/np.linalg.norm(b))*b


# In[11]:


print (vec_c)


# In[12]:


def plot_vectors(vecs):
    
    plt.quiver(0,0,100,0, scale_units ='xy',scale =1 ,angles = 'xy')
    plt.quiver(0,0,0,100, scale_units ='xy',scale =1 ,angles = 'xy')
    colors = [ 'r', 'b' , 'g' , 'y']
    i = 0 

    for vec in vecs:
        plt.quiver(vec[0], vec[1], vec[2], vec[3], scale_units ='xy', angles = 'xy' , scale =1, color = colors[ i % len (colors) ])
        i+= 1
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    


# In[13]:


vecs  = [np.array([0,0,5,6]),np.array([0,0,7,-12]),np.array([0,0,1.34196891,-2.30051813])] 


# In[14]:


plot_vectors([vecs[0], vecs[1], vecs[2]])


# In[15]:


breast_cancer = sklearn.datasets.load_breast_cancer()


# In[16]:


X = breast_cancer.data
Y = breast_cancer.target


# In[17]:


df = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)


# In[18]:


df['class'] = breast_cancer.target


# In[19]:


df.head()


# In[20]:


df.describe()


# In[21]:


df.groupby("class").mean().plot.hist()


# In[22]:


df.groupby("class").mean().plot.pie(radius = 3,subplots= "True")


# In[23]:


df["class"].value_counts()


# In[24]:


df["class"].value_counts().plot.bar()


# In[25]:


df["class"].value_counts()/len(df["class"])


# In[26]:


(df["class"].value_counts()/len(df["class"])).plot.bar()


# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


x = df.drop ("class" , axis = 1)


# In[29]:


y = df["class"]


# In[30]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.1, stratify=y, random_state=10)


# In[31]:


print (x.mean(),x_train.mean(),x_test.mean())


# In[32]:


plt.plot(x_train,'*')
plt.show()


# In[33]:


plt.plot(x_train.T,'*')
plt.xticks(rotation='vertical')
plt.show()


# In[34]:


x_train_bd = x_train.apply(pd.cut,bins=2,labels=[1,0])


# In[35]:


plt.plot(x_train_bd.T,'*')
plt.xticks(rotation='vertical')
plt.show()


# In[36]:


from sklearn.metrics import accuracy_score


# In[37]:


class Perceptron:
    def __init__(self):
        self.w = None
        self.b = None
    
    def model(self,x):
        return 1 if (np.dot(self.w,x) >= self.b) else 0
        
        
        
        
    def predict(self,X):
        Y=[]
        for x in X:
            result = self.model(x)
            Y.append(result)
        return Y    
        
        
        
    def fit(self,X,Y):
        
        
         
        self.w = np.ones(X.shape[1])
        self.b = 0
        
        for x,y in zip(X,Y):
            y_pred = self.model(x)
            if y == 1 and y_pred == 0:
                self.w = self.w + x
                self.b = self.b -1
            
            elif y == 0 and y_pred == 1:
                self.w = self.w - x
                self.b = self.b +1
            
               
                
            
            
            


# In[38]:


per= Perceptron()


# In[39]:


type(x_train)


# In[40]:


per.fit(x_train,y_train)


# In[ ]:


y_pred_train = per.predict(x_train)


# In[ ]:


plt.plot(per.w)
plt.show()


# In[ ]:


plt.plot(b)
plt.show()


# In[ ]:


print ( accuracy_score(y_pred_train,y_train))


# In[ ]:


y_pred_test = per.predict(x_test)


# In[ ]:


plt.plot(per.w)
plt.show


# In[ ]:


plt.plot(b, '')
plt.show


# In[ ]:


print( accuracy_score(y_pred_test,y_test))


# In[ ]:


X = [3,4,5,6,8]
Y = [0.268,0.73,0.952,0.994,0.999]
w = 3
b = -2


def f(w,b,x):
    return 1.0/(1.0 +np.exp(-(w*x + b)))

y_pred = []

for x in X:
    result = f(w,b,x)
    y_pred.append(result)

def rmse(x,y):
    return np.sqrt(np.mean(np.power((np.array(y)-np.array(x)),2)))
print (rmse(y_pred,Y))    
     
                


# In[ ]:


def sigmoid(x,w,b):
    return 1/(1+ np.exp(-(w*x+b)))


# In[ ]:


sigmoid (0.3,0.4,0.5)


# In[ ]:


w = 0.8
b = 0.4
X = np.linspace(-10,10,100)
Y = sigmoid (X,w,b)

plt.plot(X,Y)
plt.show()


# In[ ]:


w = np.linspace(-1,1,100)
b = np.linspace(-5,5,100)
X = np.linspace(-10,10,100)
Y = sigmoid (X,w,b)

plt.plot(X,Y)
plt.show()
plt.plot(w,Y)
plt.show()
plt.plot(b,Y)
plt.show()


# In[ ]:


plt.plot(X,Y)

plt.plot(w,Y)

plt.plot(b,Y)
plt.show()


# In[ ]:


b = 2
X = np.linspace(-10,10,100)
w = np.linspace(-1,1,100)
XX,W = np.meshgrid(X,w)
Y = sigmoid (XX,W,b)

plt.plot(X,Y)
plt.show()
plt.plot(w,Y)
plt.show()


# In[ ]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX,W, Y, cmap='YlOrRd')
ax.set_xlabel('X')
ax.set_ylabel('w')
ax.set_zlabel('Y')
ax.view_init(60, 270)
fig


# In[ ]:


def sigmoid_2d(w1,w2,b,x1,x2):
    return 1/(1+np.exp(-(w1*x1+w2*x2+b)))


# In[ ]:


sigmoid_2d(0.5,0.5,1,2,3)


# In[ ]:


w1 = 0.35
w2 = 0.3
b  = 1
X1 = np.linspace(-10,10,100)
X2 = np.linspace(-10,10,100)
XX1,XX2 =  np.meshgrid(X1,X2)
Y = sigmoid_2d(w1,w2,b,XX1,XX2)


# In[ ]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX1,XX2, Y, cmap='YlOrRd')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Y');


# In[ ]:


print(Y)


# In[ ]:


plt.plot(XX1,XX2)


# In[ ]:


ax.view_init(60, 270)
fig


# In[ ]:


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("",["red","orange","green"])
plt.contourf(XX1,XX2,Y,cmap = my_cmap, alpha = 0.8)
plt.show()


# In[ ]:


w_unknown = 0.26
b_unknown = 0.53
X = np.random.random(50)*20 - 10
Y = sigmoid (X, w_unknown,b_unknown)
plt.plot(X,Y,'*')


# In[ ]:



plt.plot(Y)


# In[ ]:


def calc_loss(X,Y,w_est,b_est):
    loss = 0
    for x,y in zip(X,Y):
        loss += (y - sigmoid(X,w_est,b_est))**2
    return loss


# In[ ]:


W = np.linspace(-1,1,100)
B = np.linspace(-1,1,100)
WW,BB = np.meshgrid(W,B)
Loss = np.zeros(WW.shape)


# In[ ]:


plt.plot(WW,BB)


# In[ ]:


plt.plot(BB)


# In[ ]:


type(BB)


# In[ ]:


for i in range (WW.shape[0]):
    for j in range (WW.shape[1]):
        Loss[i,j] = ((calc_loss(X, Y, WW[i,j], BB[i,j]))


# In[ ]:


print(WW.shape,BB.shape,Loss.shape)


# In[ ]:


from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(WW,BB,Loss,cmap='YlOrRd')
ax.set_xlabel('w_est')
ax.set_ylabel('b_est')
ax.set_zlabel('Loss')


# # CLASS SIGMOID NEURON 

# In[ ]:


class SigmoidNeuron:
    
    def __init__(self):
        self.w = None
        self.b = None
    
    
    def perceptron(self,x):
        return np.dot(x,self.w.T)+ self.b
    
    
    
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    
    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred-y) * y_pred * (1 - y_pred) * x
    
    
    def grad_b(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * (y_pred) * (1 - y_pred)
    
    
    def fit(self, X, Y, epochs=1, learning_rate=1, initialize= True):
        
        #initialize w,b
        
        if initialize:
            self.w = np.random.randn(1, X.shape[1])
            self.b = 0
            
        for i in range(epochs):
            dw =0
            db =0
            for x,y in zip(X,Y):
                dw += self.grad_w(x,y)
                db += self.grad_b(x,y)
            self.w -= learning_rate * dw
            self.b -= learning_rate * db
                

    


# In[ ]:


X = np.asarray([[2.5,2.5],[4,-1], [1,-4],[3,1.25],[2,4],[1,5]])
Y = [1,1,1,0,0,0]


# In[ ]:


sn = SigmoidNeuron()
sn.fit(X,Y,1,0.25,True)


# In[ ]:


sn.fit(X,Y,1,0.5,True)
for i in range(20):
    print(sn.w,sn.b)
    sn.fit(X,Y,1,0.5,False)
     


# In[ ]:


Y_pred = sigmoid(X,sn.w,sn.b) 
print(Y_pred)


# In[ ]:


XX,YY = np.meshgrid(X,Y_pred)


# In[ ]:


plt.scatter((X[:,0],X[:,1]),(Y_pred[:,0],Y_pred[:,1]))
plt.plot(Y)


# In[ ]:


plt.scatter(Y_pred[:,0],Y_pred[:,1])


# In[ ]:



fig = plt.figure(figsize=(6, 3.2))

ax = fig.add_subplot(111)
ax.set_title('colorMap')
plt.imshow(Y_pred)
ax.set_aspect('equal')

cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
cax.get_xaxis().set_visible(False)
cax.get_yaxis().set_visible(False)
cax.patch.set_alpha(0)
cax.set_frame_on(False)
plt.colorbar(orientation='vertical')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




