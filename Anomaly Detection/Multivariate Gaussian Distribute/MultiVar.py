import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt

x=""
y=" "

def split_train_test():
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

def calc_covar_mat(x_train,x_mean):
    x_train = x_train.reshape((x_train.shape[1],x_train.shape[0]))
    mat = x_train - x_mean
    return np.matmul(mat,np.transpose(mat))

def calc_probab(x_mean,covar_mat,x_test):
    x_test = x_test.reshape((x_test.shape[1],x_test.shape[0]))
    probab = np.zeros((x_test.shape[1],1))
    for i in range(x_test.shape[1]):
        x_test_temp = x_test[:,i]
        x_test_temp = x_test_temp.reshape((x_test_temp.shape[0],1))
        expo = (-1/2)*np.matmul(np.matmul(np.transpose(x_test_temp-x_mean),np.linalg.inv(covar_mat)),(x_test_temp-x_mean))
        probab[i,0] = np.exp(expo)/((pow(2*3.14,x_test_temp.shape[0]/2)*pow(np.linalg.det(covar_mat),1/2)))
    return probab

if __name__ == "__main__":
    df = pd.read_csv('./DataSet.csv')
    x = np.array(df.iloc[1:,0:df.shape[1]-1])
    if df.shape[1] == 2: 
        x = x.reshape((x.shape[0],1))
    y = np.array(df.iloc[1:,df.shape[1]-1])
    y = y.reshape((y.shape[0],1))
    (x_train,y_train,x_test,y_test) = split_train_test()
    x_mean = np.mean(x_train,axis = 0)
    x_mean = x_mean.reshape((x_mean.shape[0],1))
    covar_mat = calc_covar_mat(x_train,x_mean)
    print(covar_mat)
    probab_xtest = calc_probab(x_mean,covar_mat,x_test)
    print(probab_xtest)