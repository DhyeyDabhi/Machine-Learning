import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = ""
y = ""

def calc_yhat(w,b):
    #calculates yhat for the given values of w and b
    # z = wT*x + b   where, wT--> transpose of weight matrix w
    # yhat = g(z)  where, g(z) = 1/(1+e^(-z))
    # w---> (n,1) b---> (1,1) x----> (n,m)
    # n = no. of features , m = no. of training example
    global x
    z = np.matmul(np.transpose(w),x) + b
    interm = 1 + np.exp(-1*z)
    yhat = 1/interm
    return np.transpose(yhat)

def calc_cost(yhat):
    #calculates the cost for the given yhat value
    #L(yhat(i),y(i)) = -1*(y(i)*log(yhat(i)) + (1-y)*log(1-yhat(i)))
    # cost = sum i=0 --> i=m L(yhat(i),y(i))
    # yhat --> (m,1) & y --> (m,1)
    global y
    losses = -1*(np.multiply(y,np.log(yhat))+np.multiply((1-y),np.log(1-yhat)))
    cost = np.sum(losses)
    return cost

def calc_der_cost_w(yhat):
    #calculate the derivative of cost function J w.r.t 'w's i.e dws (dJ/dw)
    # dw = x*dz
    # dw = x*(yhat-y)
    # yhat-y --> (m,1) x-->(n,m) dw --> (n,1)
    global x,y
    dws = np.matmul(x,(yhat-y))
    return dws

def calc_der_cost_b(yhat):
    # calculates derivative of cost J w.r.t b i.e db(dJ/db)
    # db = sum i=0 --> i=m (yhat(i) - y(i))
    # yhat-y --> (m,1) and db --> (1,1)
    global x,y
    db = yhat-y
    return np.sum(db)


if __name__ == '__main__':
    df = pd.read_excel('./Oppurtunities.xlsx')
    x = np.array(df.iloc[1:,0])
    #change the below line for multiple features x(m,n) --> x(n,m)
    x = x.reshape((1,x.shape[0]))
    y = np.array(df.iloc[1:,1])
    y = y.reshape((y.shape[0],1))  #convert to 2D array
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
    #run the while loop until the convergence
    while cost0-cost1>=pow(10,-8):
        cost0 = cost1
        w = w - 0.001*calc_der_cost_w(yhat)/y.shape[0]
        b = b - 0.001*calc_der_cost_b(yhat)/y.shape[0]
        yhat = calc_yhat(w,b)
        cost1 = calc_cost(yhat)
        i = i+1
    print(w,b,cost1,sep=" ")
    #scatter plot the initial given (x,y)
    plt.scatter(x.reshape((x.shape[1],x.shape[0])),y)
    #plots the function predicted to fit the x,y
    x = np.linspace(0,100,1000)
    x = x.reshape((1,x.shape[0]))
    yhat = calc_yhat(w,b)
    plt.plot(x.reshape((x.shape[1],x.shape[0])),yhat)
    plt.show()
