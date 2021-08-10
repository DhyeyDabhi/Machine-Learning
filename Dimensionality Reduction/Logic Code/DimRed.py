import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normalize_data(x):
    #standardize the data i.e. making the mean of every feature to 0
    #and making the standard deviation of the every feature to 1
    #take the inout x (original data) and returns the normalized x data.
    #x --> (n,m)
    #X_normalized --> (n,m)
    x_mean = np.mean(x,axis=1,keepdims = True)
    x_std = np.std(x,axis = 1,keepdims = True)
    x_normalized = np.true_divide((x-x_mean),x_std)
    return x_normalized

def calc_svd(x):
    #first calculate the sigma matrix for given x
    #sigma = x*xT
    #where x --> (n,m) therefore sigma --> (n,n)
    #calculate Singular Value Decomposition using np.linalg.svd function
    #it returns the u and s arrays where u --> (n,n) and s --> (n,1) sized arrays
    sigma = np.dot(x,np.transpose(x))
    u, s, vh = np.linalg.svd(sigma)
    dct = {
        'U' : u,
        'S' : s
    }
    return dct

def calc_variance(x,z,Ureduce):
    #calculate the variance retained for the given reduced dimensions
    #x --> (n,m) data on which dimensionality reduction is to be applied
    #z --> (k,m) dimensionality reduction performed on x, (n,m) converted to (k,m) k<n
    #Ureduce --> (n,k) is the matrix used for calculating x_approx for the given z
    #returns a floating point value denoting its 1-retained variance
    #our goal is to get the maximum variance retained for the smaller value of k
    #i.e. choose min dimension possible with maximum variance retained typically (>=0.85)
    x_approx = np.dot(Ureduce,z)
    num = (np.sum(np.sum(np.power(x-x_approx,2),axis=0,keepdims = True),axis=1))
    denom = (np.sum(np.sum(np.power(x,2),axis=0,keepdims=True),axis=1))
    return num/denom

if __name__ == '__main__':
    df = pd.read_csv('./DataSet.csv')
    x = np.array(df.iloc[:,:df.shape[1]-1])
    if df.shape[1]==2:
        x = x.reshape((1,x.shape[0]))
    else:
        x = x.reshape((x.shape[1],x.shape[0]))
    y = np.array(df.iloc[:,df.shape[1]-1])
    y = y.reshape((1,y.shape[0]))
    # x--> (n,m) and y --> (1,m)
    x_normalized = normalize_data(x)
    svd_result = calc_svd(x_normalized)
    U = svd_result['U']
    s = svd_result['S']
    #--------------1st method (caluculate variance manually and choose k accordingly)------------------
    plt.subplot(1,2,1)
    for k in range(1,x.shape[0]):
       Ureduce = U[:,0:k]
       z = np.dot(np.transpose(Ureduce),x_normalized)
       variance = calc_variance(x_normalized,z,Ureduce)
       plt.scatter(k,variance)
    x_line = np.arange(1,x.shape[1],100)
    y_line = np.full((x_line.shape[0]),0.15)
    plt.plot(x_line,y_line)
    #----------------2nd method (use the array s to calculate the variance)-----more preferred------------------------------------
    plt.subplot(1,2,2)
    for  k in range(1,x.shape[0]):
        variance = 1-(np.sum(s[:k])/np.sum(s))
        plt.scatter(k,variance)
    plt.plot(x_line,y_line)
    plt.show()