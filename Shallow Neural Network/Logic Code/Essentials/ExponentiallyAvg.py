import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('./monthly_csv.csv')
    x = np.array(df.iloc[:,2])
    x_hat = np.zeros((x.shape[0],))
    v = 0
    B = [0.2,0.5,0.9,0.98]
    colors = plt.cm.Spectral(np.linspace(0,1,len(B)))
    for b,col in zip(B,colors):
        for i in range(x.shape[0]):
            v = (b*v + (1-b)*x[i])*(1-pow(b,i+1))
            x_hat[i] = v
        t = np.arange(1,x.shape[0]+1)
        plt.scatter(t,x)
        plt.plot(t,x_hat,color = col)
    plt.show()