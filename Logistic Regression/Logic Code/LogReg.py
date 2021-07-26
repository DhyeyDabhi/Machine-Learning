import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = ""
y = ""

def calc_yhat(w,b):
    global x
    z = np.matmul(np.transpose(w),x) + b
    interm = 1 + np.exp(-1*z)
    yhat = 1/interm
    return np.transpose(yhat)

def calc_cost(yhat):
    global y
    losses = -1*(np.multiply(y,np.log(yhat))+np.multiply((1-y),np.log(1-yhat)))
    cost = np.sum(losses)
    return cost

def calc_der_cost_w(yhat):
    global x,y
    dws = np.matmul(x,(yhat-y))
    return dws

def calc_der_cost_b(yhat):
    global x,y
    db = yhat-y
    return np.sum(db)


if __name__ == '__main__':
    df = pd.read_excel('./Oppurtunities.xlsx')
    x = np.array(df.iloc[1:,0])
    #change the below line for multiple features
    x = x.reshape((1,x.shape[0]))
    y = np.array(df.iloc[1:,1])
    y = y.reshape((y.shape[0],1))
    w = np.zeros((x.shape[0],1))
    b = np.zeros((1,1))
    yhat = calc_yhat(w,b)
    i = 0
    cost0 = calc_cost(yhat)
    w = w - 0.001*calc_der_cost_w(yhat)/y.shape[0]
    b = b - 0.001*calc_der_cost_b(yhat)/y.shape[0]
    yhat = calc_yhat(w,b)
    #print(x.shape,y.shape,yhat.shape,sep=" ")
    i = 1
    cost1 = calc_cost(yhat)
    while cost0-cost1>=pow(10,-8):
        cost0 = cost1
        w = w - 0.001*calc_der_cost_w(yhat)/y.shape[0]
        b = b - 0.001*calc_der_cost_b(yhat)/y.shape[0]
        yhat = calc_yhat(w,b)
        cost1 = calc_cost(yhat)
        i = i+1
    print(w,b,cost1,sep=" ")
    plt.scatter(x.reshape((x.shape[1],x.shape[0])),y)
    x = np.linspace(0,100,1000)
    x = x.reshape((1,x.shape[0]))
    yhat = calc_yhat(w,b)
    plt.plot(x.reshape((x.shape[1],x.shape[0])),yhat)
    plt.show()
