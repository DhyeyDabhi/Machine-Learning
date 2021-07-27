import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x=""
y=""

def calc_yhat(w,b):
    global x
    yhat = np.matmul(x,w) + b
    return yhat

def calc_cost(yhat):
    global y
    err = yhat - y
    err = np.power(err,2)
    sum_sq_err = np.sum(err)
    return sum_sq_err/y.shape[0]

def calc_der_cost_w(yhat):
    global x,y
    dws = np.matmul(np.transpose(x),(yhat-y))
    return dws

def calc_der_cost_b(yhat):
    global x,y
    db = yhat-y
    return np.sum(db)

if __name__=='__main__':
    df = pd.read_excel('./Dataset.xlsx')
    x = np.array(df.iloc[1:,0])
    #change the x incase of multiple features
    x = x.reshape((x.shape[0],1))
    y = np.array(df.iloc[1:,1])
    y = y.reshape((y.shape[0],1))
    w = np.zeros((x.shape[1],1))
    b = np.zeros((1,1))
    yhat = calc_yhat(w,b)
    #i = 0
    cost0 = calc_cost(yhat)
    #plt.scatter(i,cost0)
    w = w - 0.0001*calc_der_cost_w(yhat)
    b = b - 0.0001*calc_der_cost_b(yhat)
    yhat = calc_yhat(w,b)
    cost1 = calc_cost(yhat)
    #i = 1
    #plt.scatter(i,cost1)
    while cost0-cost1>=pow(10,-5):
        cost0 = cost1
        w = w - 0.0001*calc_der_cost_w(yhat)
        b = b - 0.0001*calc_der_cost_b(yhat)
        yhat = calc_yhat(w,b)
        #i = i + 1
        cost1 = calc_cost(yhat)
        #plt.scatter(i,cost1)
    print(w,b)
    plt.scatter(x,y)
    x = np.linspace(10,30,1000)
    x = x.reshape((x.shape[0],1))
    yhat = calc_yhat(w,b)
    plt.plot(x,yhat)
    plt.show()