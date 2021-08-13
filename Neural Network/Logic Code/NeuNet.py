import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import datetime

def sigmoid_activation(z):
    #returns the sigmoid activation values for the given z (no. of hidden units,m)
    return (1/(1+np.exp(-1*z)))

def sigmoid_der(a):
    #returns the derivative of sigmoid function for thr given activation values a (no. of hidden units,m)
    return np.multiply(a,(1-a))

def relu_activation(z):
    #returns the ReLU activation values for the given z (no. of hidden units,m)
    arr = np.maximum(z,0)
    return arr

def relu_der(a):
    #returns the derivative of ReLU function for thr given activation values a (no. of hidden units,m)
    def calc(x):
        if x:
            return 1
        else:
            return 0
    a = (a>=0)
    a = pd.DataFrame(a)
    a = a.applymap(calc)
    a = np.array(a)
    return a
    
def leaky_relu_activation(z):
    #returns the Leaky ReLU activation values for the given z (no. of hidden units,m)
    arr = np.maximum(z,0.01*z)
    return arr

def relu_der(a):
    #returns the derivative of Leaky ReLU function for thr given activation values a (no. of hidden units,m)
    def calc(x):
        if x:
            return 1
        else:
            return 0.01
    a = (a>=0)
    a = pd.DataFrame(a)
    a = a.applymap(calc)
    a = np.array(a)
    return a

def calc_cost(yhat):
    #calculates the cost value for the predicted a or yhat (1,m)
    global y
    losses = -1*(np.multiply(y,np.log(yhat))+np.multiply((1-y),np.log(1-yhat)))
    cost = np.sum(losses)
    return cost/y.shape[1]

def calc_yhat(params):
    #calculates the yhat using forward propogation, given parameters of neural network w(i),b(i)
    #w(i).shape =  (n(i),n(i-1))
    #b(i).shape = (n(i),1)
    #the function returns the tuple of output values i.e z's and a's for future use
    global x
    z1 = np.matmul(params[0],x) + params[1]
    a1 = relu_activation(z1)
    z2 = np.matmul(params[2],a1) + params[3]
    a2 = relu_activation(z2)
    z3 = np.matmul(params[4],a2) + params[5]
    a3 = sigmoid_activation(z3)
    return (z1,a1,z2,a2,z3,a3)

def calc_der(result,params):
    #calculates the derivative of parameters dw(i) ,db(i) (dimensions same as w(i) and b(i)) and returns them in form of tuple
    #use tuple returned by calc_yhat and parameters to calculate the derivatives
    #da = w(i+1)T * dz(i+1)
    #dz = da .* g'(z)       where, .* --> element wise multiplication
    #                              g'(z) ---> derivative of activation function w.r.t z
    #dw = dz*a(i-1)T
    #db = sum(dz,axis = 1)
    global x,y
    dz3 = result[5] - y
    dw3 = np.matmul(dz3,np.transpose(result[3]))/y.shape[1]
    db3 = np.sum(dz3,axis=1,keepdims=True)/y.shape[1]
    dz2 = np.multiply(np.matmul(np.transpose(params[4]),dz3),relu_der(result[3]))
    dw2 = np.matmul(dz2,np.transpose(result[1]))/y.shape[1]
    db2 = np.sum(dz2,axis=1,keepdims=True)/y.shape[1]
    dz1 = np.multiply(np.matmul(np.transpose(params[2]),dz2),relu_der(result[1]))
    dw1 = np.matmul(dz1,np.transpose(x))/y.shape[1]
    db1 = np.sum(dz1,axis=1,keepdims=True)/y.shape[1]
    return (dw1,db1,dw2,db2,dw3,db3)

def update_params(params,params_der):
    #update the parameters as per the grad desc. algo
    #learning rate = 0.01 , here assumed it 0.01 but should run a hyperparameter tuning process
    w1 = params[0] - 0.01*params_der[0]
    b1 = params[1] - 0.01*params_der[1]
    w2 = params[2] - 0.01*params_der[2]
    b2 = params[3] - 0.01*params_der[3]
    w3 = params[4] - 0.01*params_der[4]
    b3 = params[5] - 0.01*params_der[5]
    return (w1,b1,w2,b2,w3,b3)

if __name__=="__main__":
    df = pd.read_excel('./Opportunity.xlsx')
    x = np.array(df.iloc[:,0])
    x = x.reshape((1,x.shape[0]))
    y = np.array(df.iloc[:,1])
    y = y.reshape((1,y.shape[0]))
    w1 = np.random.randn(4,x.shape[0])/10
    b1 = np.zeros((4,1))
    w2 = np.random.randn(4,4)/10
    b2 = np.zeros((4,1))
    w3 = np.random.rand(1,4)/10
    b3 = np.zeros((1,1))
    params = (w1,b1,w2,b2,w3,b3)
    time1 = datetime.datetime.now()
    #-------------------------------------------
    result = calc_yhat(params)
    cost0 = calc_cost(result[5])
    params_der = calc_der(result,params)
    params = update_params(params,params_der)
    #--------------------------------------------
    result = calc_yhat(params)
    cost1 = calc_cost(result[5])
    while cost1>=0.4:
        params_der = calc_der(result,params)
        prev_params = params
        params = update_params(params,params_der)
        result = calc_yhat(params)
        cost0 = cost1
        cost1 = calc_cost(result[5])
    time2 = datetime.datetime.now()
    #plot the x's and y's given
    plt.scatter(x.reshape((x.shape[1],x.shape[0])),y.reshape((y.shape[1],y.shape[0])))
    x = np.linspace(0,100,1000)
    #arrange x at equal intervals and calculate yhat at the respective x for plotting the curve
    x = x.reshape((1,x.shape[0]))
    yhat = calc_yhat(prev_params)
    yhat = np.array(yhat[5])
    plt.plot(x.reshape((x.shape[1],x.shape[0])),yhat.reshape((yhat.shape[1],yhat.shape[0])))
    plt.show()
    print(time2-time1,cost1,sep=" ")