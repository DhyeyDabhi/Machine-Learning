import pandas as pd
import numpy as np
import random
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

x= ""

def calc_color_indxs(centroids):
    #function assigns centroid indexes to each training example i.e. it assigns the 
    # nearest cluster centroid to each training example
    # It uses Eucledian Distance to measure the distance between cluster centroids and training example
    global x
    centroid_indx = np.zeros(((x.shape[0]),1))
    for i in range(0,x.shape[0]):
        dist = x[i,:]-centroids
        dist = np.sum(np.power(dist,2),axis = 1)
        centroid_indx[i] = np.argmin(dist)
    return centroid_indx.astype(int)

def calc_cost(centroids,sample_color_indx):
    #calculates the cost value of the calculated centroid.
    #cost = average of the distances between the centroids and the assigned training examples
    sample_centroids = centroids[sample_color_indx.reshape((sample_color_indx.shape[0]))]
    dist = x - sample_centroids
    dist = np.sum(np.power(np.sum(np.power(dist,2),axis = 1),0.5),axis = 0)
    return dist/sample_centroids.shape[0]

def update_centroids(centroids,sample_color_indx,k):
    #updates the centroid for each assigned cluster
    #calculates the centroid by taking mean of all the example assigned to the cluster
    for i in range(0,k):
        indxs = np.where(sample_color_indx == i)
        x_centroid = x[indxs[0]]
        if x_centroid.shape[0] == 0:
            continue
        centroids[i] = np.mean(x_centroid,axis = 0)
    return centroids

if __name__ == '__main__':
    data = load_iris(as_frame = True)
    df = data.data
    num_of_features = df.shape[1] 
    x = np.array(df.iloc[1:,0:num_of_features])
    k = int(input("Enter Number of Clusters: "))
    random_init_indx = random.sample(range(0,df.shape[0]),k)
    centroids = np.array(df.iloc[random_init_indx,0:num_of_features])
    plt.subplot(1,2,1)
    i = 0
    #------------------------------------------------------------------
    sample_color_indx = calc_color_indxs(centroids) #step1
    cost0 = calc_cost(centroids,sample_color_indx)
    prev_centroids = centroids
    centroids = update_centroids(centroids,sample_color_indx,k) #step2\
    plt.scatter(i,cost0)
    i = i + 1
    #----------------------------------------------------------------
    sample_color_indx = calc_color_indxs(centroids) #step1
    cost1 = calc_cost(centroids,sample_color_indx) #step2
    #--------------------------------------------------------------------
    while cost0-cost1>=pow(10,-9):
        i = i + 1
        plt.scatter(i,cost1)
        prev_centroids = centroids
        centroids = update_centroids(centroids,sample_color_indx,k)
        cost0 = cost1
        sample_color_indx = calc_color_indxs(centroids)
        cost1 = calc_cost(centroids,sample_color_indx)
    print(cost0)
    #plots two subplots in a figure,
    #1.) Cost funcn vs. no. of iterations 
    #2.) Plot Training examples of same clusters with same color.
    plt.subplot(1,2,2)
    sample_color_indx = calc_color_indxs(prev_centroids)
    colors = plt.cm.Spectral(np.linspace(0,1,k))
    for i,col in zip(range(k),colors):
        indxs = np.where(sample_color_indx == i)
        x_centroid = x[indxs[0]]
        plt.scatter(x_centroid[:,0],x_centroid[:,1],color = col)
    plt.show()
