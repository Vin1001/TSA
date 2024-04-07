import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns

class LinearRegress:
    #multivar linear regression
    def __init__(self, df:pd.DataFrame, *cols):
        #if len(col)==2:
            self.new_mat = pd.DataFrame()
            self.columns = np.array(cols)
            for col in cols:
                self.new_mat[df.iloc[:,col-1:col].columns] = np.array(df.iloc[:,col-1:col])
                
            self.m = self.new_mat.shape
            self.X_train = pd.DataFrame()
            self.y_train = pd.DataFrame()
            self.U = np.empty([self.m[1],self.m[1]], dtype=float)
            self.V = np.empty([self.m[1],1], dtype=float)
            self.W = np.empty([self.m[1],1], dtype=float)
            
        #else:
            #print("Enter two Columns")
            
    def show(self):
        return self.new_mat

    def target(self, target):
        
        self.y_train[self.new_mat.iloc[:,target-1:target].columns] = np.array(self.new_mat.iloc[:,target-1:target])
        #display(self.y_train)
        self.X_train = self.new_mat.drop(self.y_train.columns, axis=1)
        #self.X_train['1'] = 1 
        sums = np.array([])
        for col in np.delete(np.arange(self.m[1]), target-1):
            sums = np.append(sums, np.sum(self.new_mat.iloc[:,col]))
            
        trgt_sum = np.sum(self.new_mat.iloc[:,target-1])  #here
        
        for i in np.arange(len(sums)):
            self.U[0][i] = sums[i]
            
        self.U[0][-1] = self.new_mat.iloc[:,target-1].count()
        
        self.V[0][0] = trgt_sum
        
        for i in np.arange(len(sums)):
            for j in np.arange(len(sums)):
                self.U[i+1][j] = np.sum(self.X_train.iloc[:,j]/self.X_train.iloc[:,i])
                
            self.U[i+1][-1] = np.sum(1/self.X_train.iloc[:,i])
            self.V[i+1][0] = np.sum(self.y_train.iloc[:,0]/self.X_train.iloc[:,i])   #here
                
        self.W = la.inv(self.U) @ self.V
        #print(self.W)
        
    def predict(self, *test_data):
        pred_val = 0
        i = 0
        for value in test_data:
            pred_val += self.W[i][0]*value
            #print(self.W[i][0],value,pred_val)
            i+=1
        
        return pred_val + self.W[-1][0]
    
    def plot(self, sz=10, wd=1):
        if self.X_train.shape[1]==1:
            ledgy = list()
            y_pred = np.array([])
            for i in np.arange(self.m[0]):
                 y_pred = np.append(y_pred, self.predict(*tuple(self.X_train.iloc[i,:])))
                
            for col in np.arange(1,self.m[1]):
                plt.scatter((self.X_train.iloc[:,col-1:col]), (self.y_train), color=(np.random.random(),np.random.random(),np.random.random()), s=sz)
                ledgy.append(self.X_train.iloc[:1,col-1:col].columns[0])
            
            plt.plot((self.X_train.iloc[:,:1]), (y_pred), color=(np.random.random(),np.random.random(),np.random.random()), linewidth=wd+0.5)
            plt.legend(ledgy, loc=(1,0.7))
            plt.ylabel(self.y_train.iloc[:1,:1].columns[0])
            plt.show()
        else:
            return "Please Enter one on one relation"
    
    def accuracy(self):
        if self.X_train.shape[1]>=-1:
            y_pred = np.array([])
            for i in np.arange(self.m[0]):
                y_pred = np.append(y_pred, self.predict(*tuple(self.X_train.iloc[i,:])))
                
            #acc = 100 - ((abs(np.reshape(np.array(self.y_train), (self.m[0],))-y_pred)/np.reshape(np.array(self.y_train), (self.m[0],)))*100)
            #print(len(abs(np.reshape(np.array(self.y_train), (22,))-y_pred)/np.reshape(np.array(self.y_train), (22,))))
            acc = abs(np.reshape(np.array(self.y_train), (self.m[0],))-y_pred)
            return acc.mean()
        else:
            return "Please Enter one on one relation"
        
    def error(self,perc=80):
        if self.X_train.shape[1]>=-1:
            y_pred = np.array([])
            for i in np.arange(self.m[0]):
                y_pred = np.append(y_pred, self.predict(*tuple(self.X_train.iloc[i,:])))
                
            #acc = 100 - ((abs(np.reshape(np.array(self.y_train), (self.m[0],))-y_pred)/np.reshape(np.array(self.y_train), (self.m[0],)))*100)
            #print(len(abs(np.reshape(np.array(self.y_train), (22,))-y_pred)/np.reshape(np.array(self.y_train), (22,))))
            acc = abs(np.reshape(np.array(self.y_train), (self.m[0],))-y_pred)
            # sns.boxplot(acc)
            # plt.show()
            sns.histplot(acc, kde=True)
            plt.show()
            print(f'mean:{np.mean(acc)},std:{np.std(acc)}')
            return np.percentile(acc,perc)
        else:
            return "Please Enter one on one relation"
        

class LinearRegression:
    #multivar linear regression
    def __init__(self, df, *cols):
        #if len(col)==2:
            self.new_mat = pd.DataFrame()
            self.columns = np.array(cols)
            for col in cols:
                self.new_mat[df.iloc[:,col-1:col].columns] = np.array(df.iloc[:,col-1:col])
                
            self.m = self.new_mat.shape
            self.X_train = pd.DataFrame()
            self.y_train = pd.DataFrame()
            self.U = np.empty([self.m[1],self.m[1]], dtype=float)
            self.V = np.empty([self.m[1],1], dtype=float)
            self.W = np.empty([self.m[1],1], dtype=float)
            
        #else:
            #print("Enter two Columns")
            
    def show(self):
        print(self.new_mat)
        
    def target(self, target):
        
        self.y_train[self.new_mat.iloc[:,target-1:target].columns] = np.array(self.new_mat.iloc[:,target-1:target])
        #display(self.y_train)
        self.X_train = self.new_mat.drop(self.y_train.columns, axis=1)
        #self.X_train['1'] = 1 
        sums = np.array([])
        for col in np.delete(np.arange(self.m[1]), target-1):
            sums = np.append(sums, np.sum(self.new_mat.iloc[:,col]))
            
        trgt_sum = np.sum(self.new_mat.iloc[:,target-1])  #here
        
        for i in np.arange(len(sums)):
            self.U[0][i] = sums[i]
            
        self.U[0][-1] = self.new_mat.iloc[:,target-1].count()
        
        self.V[0][0] = trgt_sum
        
        for i in np.arange(len(sums)):
            for j in np.arange(len(sums)):
                self.U[i+1][j] = np.sum(self.X_train.iloc[:,j]/self.X_train.iloc[:,i])
                
            self.U[i+1][-1] = np.sum(1/self.X_train.iloc[:,i])
            self.V[i+1][0] = np.sum(self.y_train.iloc[:,0]/self.X_train.iloc[:,i])   #here
                
        self.W = la.inv(self.U) @ self.V
        #print(self.W)
        
    def predict(self, *test_data):
        pred_val = 0
        i = 0
        for value in test_data:
            pred_val += self.W[i][0]*value
            #print(self.W[i][0],value,pred_val)
            i+=1
        
        return pred_val + self.W[-1][0]
    
    def plot(self, sz=10, wd=1):
        if self.X_train.shape[1]==1:
            ledgy = list()
            y_pred = np.array([])
            for i in np.arange(self.m[0]):
                 y_pred = np.append(y_pred, self.predict(*tuple(self.X_train.iloc[i,:])))
                
            for col in np.arange(1,self.m[1]):
                plt.scatter((self.X_train.iloc[:,col-1:col]), (self.y_train), color=(np.random.random(),np.random.random(),np.random.random()), s=sz)
                ledgy.append(self.X_train.iloc[:1,col-1:col].columns[0])
            
            plt.plot((self.X_train.iloc[:,:1]), (y_pred), color=(np.random.random(),np.random.random(),np.random.random()), linewidth=wd+0.5)
            plt.legend(ledgy, loc=(1,0.7))
            plt.ylabel(self.y_train.iloc[:1,:1].columns[0])
            plt.show()
        else:
            return "Please Enter one on one relation"
    
    def error(self,perc=80):
        if self.X_train.shape[1]>=-1:
            y_pred = np.array([])
            for i in np.arange(self.m[0]):
                y_pred = np.append(y_pred, self.predict(*tuple(self.X_train.iloc[i,:])))
                
            #acc = 100 - ((abs(np.reshape(np.array(self.y_train), (self.m[0],))-y_pred)/np.reshape(np.array(self.y_train), (self.m[0],)))*100)
            #print(len(abs(np.reshape(np.array(self.y_train), (22,))-y_pred)/np.reshape(np.array(self.y_train), (22,))))
            acc = abs(np.reshape(np.array(self.y_train), (self.m[0],))-y_pred)
            #sns.boxplot(acc)
            #plt.show()
            sns.histplot(acc, kde=True)
            plt.show()
            print(f'mean:{np.mean(acc)},std:{np.std(acc)}')
            return np.percentile(acc,perc)
        else:
            return "Please Enter one on one relation"
        
def main():
    print("Initialize the object using whole dataframe and surrogate column numbers(1,2,3,4...) \nor see the full implementation at https://github.com/Vin1001/TSA/blob/main/notebooks/autoreg%20.ipynb")

if __name__ == "__main__":
    
    main()