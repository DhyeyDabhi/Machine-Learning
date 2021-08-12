import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''The method of calculating moving average is called Exponentially Weighted Average.
    Moving a window with the two extreme endpoints coulld work but  it would require more memory space to carry out 
    the process. As well as thiss ,ethod would be easier to implement while implementing Momentum Grad Desc and
    RMSprop Grad desc.
    In this method we calculate the averages over the last 1/(1-B) examples where B is the one of the hyper parameter 
    to  be tuned.
    Hyper Parammeters : B'''
    df = pd.read_csv('./monthly_csv.csv')
    x = np.array(df.iloc[:,2])
    B = [0.2,0.5,0.9,0.98]
    # Therefore respective number of exmples over which average is taken: 
    # 1/(1-B) = [1.25,2,10,50]
    colors = plt.cm.Spectral(np.linspace(0,1,len(B)))
    j = 1
    for b,col in zip(B,colors):
        x_hat = np.zeros((x.shape[0],))
        v = 0
        for i in range(x.shape[0]):
            v = (b*v + (1-b)*x[i])
            x_hat[i] = v
        t = np.arange(1,x.shape[0]+1)
        plt.subplot(2,2,j)
        plt.scatter(t,x)
        plt.plot(t,x_hat,color = col)
        j = j+1
    plt.show()
    '''Figure will show 4 different plots having different hyper parameter B 
    You would notice that plot with less B is having very noisy curve as it takes into account less number of 
    Examples. Therefore, the hyper parameter should be chosen according to the need of the algorithm'''
    