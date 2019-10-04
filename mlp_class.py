import numpy as np

class NET ():
    def __init__ (self, hidden, epoch, eps, eta, func_hidden, func_out, build_graph):
        self.hidden = hidden
        self.epoch = epoch
        self.eps = eps
        self.eta = eta
        self.func_hidden = func_hidden
        self.func_out = func_out
        self.build_graph = build_graph
        self.flag = False
        
    def targets (self,y_train):
        k = y_train.nunique()
        n = y_train.shape[0] 
        y = y_train.values.reshape(-1,1).astype('int') 
        d_all = np.zeros((n,k)) 
        for i in range (len(y)): 
            d_all[i][y[i]] = 1 
        return d_all
        
    def standartize (self,X_train):
        self.mean_train = X_train.mean()
        self.std_train = X_train.std()
        X_train = (X_train-self.mean_train)/self.std_train
        return X_train
    
    def graph (self,d):
        plt.figure(figsize = (6,6)) 
        plt.title ('MSE(epoch)') 
        plt.xlabel('epoch')
        plt.ylabel('MSE')
        plt.plot(d.keys(),d.values())
        
    def func(self, x , tip , deriv): 
        if tip  == 'sigm' and deriv == False:
            return 1/(1+np.exp(-x))
        elif tip == 'sigm' and deriv == True:
            return (1/(1+np.exp(-x)))*(1 - 1/(1+np.exp(-x)))
        elif tip == 'tanh' and deriv == False:
            return np.tanh(x)
        elif tip == 'tanh' and deriv == True:
            return 1 - (np.tanh(x))**2
        else:
            print('Something was wrong')
            return None
    
    def fit(self, X_train, y_train):
        d_all = self.targets(y_train)
        X_train = self.standartize(X_train)
        self.w1 = np.random.rand(self.hidden,X_train.shape[1])
        self.w2 = np.random.rand (y_train.nunique(),self.hidden)
        mse = 0  
        d ={}
        for i in range (self.epoch): 
            seq = list(range(X_train.shape[0]))
            np.random.shuffle(seq) 
            for j in seq :
                x = X_train.values[j].reshape (X_train.shape[1],1) 
                s1 = self.w1@x
                func_s1 = self.func(s1,self.func_hidden,deriv = False)
                s2 = self.w2@func_s1
                func_s2 = self.func(s2,self.func_out, deriv = False) 
                mse += np.sum ((func_s2-d_all[j].reshape(-1,1))**2)/2 
                # алгоритм обратного распостранения ошибки
                delta_2 = (func_s2-d_all[j].reshape(-1,1))*self.func(s2,self.func_out,deriv = True) 
                delta_1 = self.w2.T@delta_2*self.func(s1,self.func_hidden,deriv = True) 
                self.w2 -= self.eta * delta_2@func_s1.T 
                self.w1 -= self.eta * delta_1@x.T 
                
            d[i] = mse 
            if mse < self.eps:
                self.flag = True
                if self.build_graph :
                    self.graph(d)
                return None
            mse = 0
        self.flag = True
        if self.build_graph:
            self.graph(d)
        return None
    
    def predict (self,df):
        if not self.flag:
            raise Exception('Модель ещё не обучена')
        df = (df - self.mean_train) / self.std_train
        res = []
        res_proba = []
        for i in range(df.shape[0]) :
            x = df.values[i].reshape (df.shape[1],1)
            func_s1 = self.func(self.w1@x,self.func_hidden,deriv = False) 
            func_s2 = self.func(self.w2@func_s1,self.func_out,deriv = False)
            res_proba += [max(func_s2)]
            res += [np.argmax(func_s2)]
        return np.array(res)
            
            
    def quality (self,y_pred,y_true):
        try:
            print('CONFUSION MATRIX:')
            print(np.array([[metrics.confusion_matrix(y_pred,y_true)[1][1],
                  metrics.confusion_matrix(y_pred,y_true)[0][1]],
                  [metrics.confusion_matrix(y_pred,y_true)[1][0],
                  metrics.confusion_matrix(y_pred,y_true)[0][0]]]))
            print ('METRICS:')
            print('accuracy_ratio:', metrics.accuracy_score(y_pred,y_true))
            print('accuracy_count:',metrics.accuracy_score(y_pred,y_true,normalize = False))
            print('balanced_accuracy:',metrics.balanced_accuracy_score(y_pred,y_true))
            print('precision:',metrics.precision_score(y_pred,y_true))
            print('recall:',metrics.recall_score(y_pred,y_true))
            print('f1_score:',metrics.f1_score(y_pred,y_true))
        except Exception:
            print('Something was wrong')
        return metrics.recall_score(y_pred,y_true)