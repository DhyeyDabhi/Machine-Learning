import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = ""
y = ""

def split_train_test():
    #split the data set into training and testing data
    #the splitting is done by classifying training set having y=0 i.e all non-anomalous data points
    #and splitting into test set having y=1 i.e all anomalous data points
    global x,y
    bool_arr = (y==0)
    bool_arr = bool_arr.reshape((x.shape[0]))
    x_train = x[bool_arr]
    y_train = y[bool_arr]
    bool_arr = (y==1)
    bool_arr = bool_arr.reshape((x.shape[0]))
    x_test = x[bool_arr]
    y_test = y[bool_arr]
    return (x_train,y_train,x_test,y_test)

def calc_probab(x_test,x_mean,x_std_dev):
    #calculate the probability of occurence of the given data-point
    #mean and standard deviation are provided
    # p(x) = mul i=1 --> i=n p(xi;ui,si)
    # where xi is the ith feature of the x sample
    # ui ---> mean of the ith feature
    # si ---> stanndard deviation of the ith feature
    global x
    expo = -1*np.divide(np.power((x_test-x_mean),2),2*np.power(x_std_dev,2))
    probab = np.divide(np.exp(expo),2.506628*x_std_dev)
    probab = np.prod(probab,axis = 1,keepdims = True)
    return probab

if __name__ == '__main__':
    df = pd.read_csv('./DataSet.csv')
    x = np.array(df.iloc[1:,0:df.shape[1]-1])
    if df.shape[1] == 2: 
        x = x.reshape((x.shape[0],1))
    y = np.array(df.iloc[1:,df.shape[1]-1])
    y = y.reshape((y.shape[0],1))
    (x_train,y_train,x_test,y_test) = split_train_test()
    x_mean = np.mean(x_train,axis = 0)
    x_mean = x_mean.reshape((1,x_mean.shape[0]))
    x_std_dev = np.std(x_train,axis = 0)
    x_std_dev = x_std_dev.reshape((1,x_std_dev.shape[0]))
    print(x_mean.shape,x_std_dev.shape)
    x_test_probab = calc_probab(x_test,x_mean,x_std_dev)
    #classify the given x if the probability of occurence is  less than 10^(-20)
    classify_x_test = (x_test_probab<pow(10,-20))
    print(classify_x_test)