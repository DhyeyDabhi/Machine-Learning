import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x=""
y=""

#Calculate yhat (predicted y) using w and b as input parameters
def calc_yhat(w,b):
    global x
    yhat = np.matmul(x,w) + b
    return yhat

#Calculate the cost value (Mean Squarred Error) for the calculated yhat (predicted y)
def calc_cost(yhat):
    global y
    err = yhat - y
    err = np.power(err,2)
    sum_sq_err = np.sum(err)
    return sum_sq_err/y.shape[0]

#Calculate the dws(dJ/dw) for the calculated yhat, in order to update the dw values
def calc_der_cost_w(yhat):
    global x,y
    dws = np.matmul(np.transpose(x),(yhat-y))
    return dws

#Calculate the db(dJ/db) for the calculated yhat, in order to update the db values
def calc_der_cost_b(yhat):
    global x,y
    db = yhat-y
    return np.sum(db)

#Here, x has shape (m,n) where, n = num of features, m = np. of training example
#      y,yhat has shape (m,1)
#      w,dws has shape (n,1)
#      b,db has shape (1,1)
if __name__=='__main__':
    df = pd.read_excel('./Dataset.xlsx')
    x = np.array(df.iloc[1:,0])
    #change the x incase of multiple features
    x = x.reshape((x.shape[0],1))
    y = np.array(df.iloc[1:,1])
    y = y.reshape((y.shape[0],1))
    w = np.zeros((x.shape[1],1))
    b = np.zeros((1,1))
    #----------------------------------
    yhat = calc_yhat(w,b)
    #i = 0
    cost0 = calc_cost(yhat)
    #plt.scatter(i,cost0)
    w = w - 0.0001*calc_der_cost_w(yhat)
    b = b - 0.0001*calc_der_cost_b(yhat)
    #----------------------------------
    yhat = calc_yhat(w,b)
    cost1 = calc_cost(yhat)
    #i = 1
    #plt.scatter(i,cost1)
    w = w - 0.0001*calc_der_cost_w(yhat)
    b = b - 0.0001*calc_der_cost_b(yhat)
    #------------------------------------
    while cost0-cost1>=pow(10,-5):
        cost0 = cost1
        yhat = calc_yhat(w,b)
        #i = i + 1
        cost1 = calc_cost(yhat)
        #plt.scatter(i,cost1)
        w = w - 0.0001*calc_der_cost_w(yhat)
        b = b - 0.0001*calc_der_cost_b(yhat)  
    print(w,b)
    plt.scatter(x,y)
    x = np.linspace(10,30,1000)
    x = x.reshape((x.shape[0],1))
    yhat = calc_yhat(w,b)
    plt.plot(x,yhat)
    plt.show()