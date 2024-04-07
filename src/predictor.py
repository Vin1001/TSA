import mypackages.Fitter as fit
import yfinance as yf
import pandas as pd
import numpy as np
import numpy.linalg as la
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt



class stock_predictor:


    def __init__(self, name, per, intrvl, cols = ['Close']) -> None:
        df = yf.Ticker(name)
        df = pd.DataFrame(df.history(period=per, interval=intrvl))
    
        self.df1 = df[cols].copy()
        plot_pacf(self.df1, lags=30);
        self.df_train : pd.DataFrame
        self.df_test : pd.DataFrame
        self.intv : float
        self.dev = np.std(self.__minMax(self.df1.tail(10), 0 , 1))


    def lags(self, n_lags):
        temp = [self.df1]+[self.df1.shift(-i) for i in np.arange(1,n_lags+1)]

        self.cols = temp

        dff = pd.concat(self.cols, axis=1)
        #dff = pd.concat([df1,df1['Close'].shift(-1),df1['Close'].shift(-2),df1['Close'].shift(-3),df1['Close'].shift(-4),df1['Close'].shift(-5)], axis=1)
        dff.columns = np.arange(len(self.cols))
        dff = dff.dropna()
        size = dff.shape[0]-10
        self.df_train = dff.iloc[:size,:]
        self.df_test = dff.iloc[size:,:]

        
        return dff.tail(3)
    
    
    def fit(self, perc=80):

        self.lr = fit.LinearRegression(self.df_train, *np.arange(1,len(self.cols)+1))
        self.lr.target(np.arange(1,len(self.cols)+1)[-1])
        l = list()
        for i in np.arange(self.df_test.shape[0]):
            l.append(self.lr.predict(*(self.df_test.iloc[i,:-1])))
        
        self.l=np.array(l)
        self.intv = self.lr.error(perc)
        #print(self.l)

        return self.intv
    
    def plot_predict(self):

        self.__plot_predict(np.append(self.l,np.array((self.__forec(3187.95,6,t=-2,read=False)[:,2]))),self.intv)  
        #print("Hello")  
    
    def __plot_predict(self, l, intv):
        plt.figure(figsize=(5,3.8))
        lower,upper = l-intv, l+intv
        #actual = np.arange(-len(self.l),len(np.array(self.df_test.iloc[:,-1])))
        #print(actual)
        plt.plot(np.array(self.df_test.iloc[:,-1]), c='k', alpha=0.4) #np.array(df_test.iloc[:,-1])
        #print(np.array(df_test.iloc[:,-1]))
        plt.plot(lower[1:],'r--', label='lower')
        plt.plot(l[1:], c='b',label = 'predicted')
        plt.plot(upper[1:],'g--', label='upper')
        plt.legend(loc='upper left')
        plt.show()


    def forecast(self, t, d, k, look):
        prev_days = np.array(self.df_test.iloc[t,:-1])
        _ = self.__forec_tree(prev_days,float(self.df_test.iloc[t,-1]),
           self.intv,
           depth=d,
           fluctFact= self.dev*k,
           look=look)
        
        print(str(_)+'\n')
        
        self.__forec(3187.95,n_points=d,t=t )
        
        
    def __forec_tree(self, prev_days,price, intv, depth, look, fluctFact=0.25, dr=1):
        udr = 0.8
        ldr = 1.25
        prev_days = np.append(prev_days, price)
        prev_days = prev_days[1:]
        #f = np.array([prev_days[i]-prev_days[i-1 if i>0 else i] for i in np.arange(len(prev_days))])
        #f = (f + self.intv)/self.intv
        #print(f)
        m = m1 = self.lr.predict(*prev_days)
        #prev_days = np.append(prev_days, m)
        up,low = m+self.intv,m-self.intv
        disth = dr*abs(price-up)
        distl = abs(price-low)
        factor = (distl/(distl+disth))
        if depth==-look:
            #print("in -5")
            return factor
        
        if depth<=0 and depth>-look:
            u = self.__forec_tree(prev_days, price+(fluctFact)*1*intv, intv, depth-1, fluctFact= fluctFact, dr=1 if fluctFact<=0.7 or fluctFact >= 1.2 else udr*dr,look=look)
            d = self.__forec_tree(prev_days, price+(fluctFact)*1*intv, intv, depth-1, fluctFact= fluctFact, dr=1 if fluctFact<=0.7 or fluctFact >= 1.2 else ldr*dr, look=look)
            #print(f"IF {price:7.6} THEN:{factor:5.2},up:{up:7.6},mean:{m1:7.6},low:{low:7.6} d({depth-1})\n")
            return (factor*u*d)**(1/3)
    
        elif depth>0:
            d = self.__forec_tree(prev_days,price-(fluctFact)*1*intv, intv, depth-1, fluctFact=fluctFact,dr=1 if fluctFact<=0.7 or fluctFact >=1.2 else ldr*dr, look=look)
            #print(f"IF {price:7.6} THEN:{factor:5.2},up:{up:7.6},mean:{m1:7.6},low:{low:7.6} d({depth-1})\n")
            u = self.__forec_tree(prev_days,price+(fluctFact)*1*intv, intv, depth-1, fluctFact=fluctFact,dr=1 if fluctFact<=0.7 or fluctFact >= 1.2 else udr*dr, look=look)
            # factor = (factor*d*u)**(1/3)
            print(f"IF {price:7.6} THEN: {factor:5.2},up:{up:7.6},mean:{m1:7.6},low:{low:7.6} d({depth-1})\n")
            return factor
        

    def __forec(self, price_today, n_points, t=-1, read=True):
        price_today = float(self.df_test.iloc[t,-1])
        prev_days = np.append(np.array(self.df_test.iloc[t,:-1]), price_today)
        prev_days = prev_days[1:]
        output = np.empty((n_points,4))
    #print(prev_days)
        for i in np.arange(n_points):
            m = m1 = self.lr.predict(*prev_days)
            up,low = m+self.intv,m-self.intv
            disth = abs(price_today-up)
            distl = abs(price_today-low)
            factor = distl/(distl+disth)
            output[i][0] = factor
            output[i][1] = up
            output[i][2] = m1
            output[i][3] = low
            '''if factor<0.47:
                m = price_today = (m+(factor*.001*m))
            else:
                m = price_today = (m-factor*.001*m)'''
            price_today = m
            prev_days = np.append(np.array(prev_days),price_today)
            prev_days = prev_days[1:]

            if read:
                print(f'[{factor:5.2}, up: {up:7.6}, mean: {m1:7.6}, low: {low:7.6}], {float(self.df_test.iloc[t,-1]):7.6}')

        if read==False:
            return output
        

    def __minMax(self, df, lower, upper):
        vec = np.array(df)
        # print(len(df.shape))
        for j in np.arange(vec.shape[1] if len(vec.shape)>1 else 1):
            min = np.min(vec[:,j] if len(vec.shape)>1 else vec[j])
            max = np.max(vec[:,j] if len(vec.shape)>1 else vec[j])
            for i in np.arange(vec.shape[0]):
                
                if len(vec.shape)>1:
                    vec[i,j] = ((vec[i,j] - min)/(max-min))*(upper-lower)+lower

                else:
                    vec[i] = ((vec[i] - min)/(max-min))*(upper-lower)+lower


        return vec