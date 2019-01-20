import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class DTW():
    def __init__(self,X,Y,metric=euclidean_distances,step=120):
        self.X = X
        self.Y = Y
        self.metric = metric
        self.len_x, self.len_y = len(X), len(Y)
        self.step = step
        
    def _dtw(self,X,Y,metric=euclidean_distances,early_stopping=False):
        self._len_x,self._len_y = X.shape[0]-1, Y.shape[0]-1
        I, J = 0, 0
        path = [(0,0)]
        if len(X.shape) == 1 or len(Y.shape) == 1:
            self.cost_matrix = metric(X.reshape((-1,1)),Y.reshape((-1,1)))
        else:
            self.cost_matrix = metric(X,Y)
        distances = [self.cost_matrix[I,J]]
        
        ## cythonize below??
        for n in range(self._len_x+self._len_y): 
            # check boundary conditions
            if I >= self._len_x and J < self._len_y:
                if early_stopping:
                    break
                distance = [self.cost_matrix[I,J+1]] # to match length
                direction = 0 # compulsory direction
            elif J >= self._len_y and I < self._len_x:
                if early_stopping:
                    break
                distance = [-1,self.cost_matrix[I+1,J]] # to match length
                direction = 1 # compulsory direction
            elif I >= self._len_x and J >= self._len_y:
                break
            else:
                distance = [self.cost_matrix[I,J+1], # to right
                            self.cost_matrix[I+1,J], # to above
                            self.cost_matrix[I+1,J+1]] # to right and above
                direction = np.argmin(distance)
                
            # set new center point
            if direction == 0: # to right
                path.append((I,J+1))
                J += 1
            elif direction == 1: # to above
                path.append((I+1,J))
                I += 1
            else: # to right and above
                path.append((I+1,J+1))
                I += 1
                J += 1
            distances.append(distance[direction])
        return np.asarray(path),np.asarray(distances),np.sum(distances)
    
    def plot(self):
        """
        Plot the mapping.
        """
        plt.style.use('ggplot')
        shift=np.std(self.Y)*2
        fig = plt.figure()
        ax = plt.axes()
        x = np.arange(self.len_x)
        y = np.arange(self.len_y)
        ax.plot(x, self.X)
        ax.plot(y, self.Y+shift)
        for i,j in self.path:
            ax.plot(np.r_[x[i],y[j]], np.r_[self.X[i],self.Y[j]+shift],ls='dashed',c='gray')
        plt.figure(figsize=(12,5))
        plt.show(ax)
        
    def __repr__(self):
        st = '...' if len(self.distances) > 3 else ''
        return 'Path : {}{}\nDistances : {}{}\nTotal :{}'.format(self.path[:3],
                       st,self.distances[:3].round(4),st,self.total.round(4))


class dtw(DTW):
    """
    Dynamic time warping calculation.
    """
    def __init__(self,X,Y,metric=euclidean_distances,step=120):
        DTW.__init__(self,X,Y,metric,step)
        if self.step is None:
            self.path,self.distances,self.total = self._dtw(self.X,self.Y,self.metric)
        else:
            self.step = self.step if abs(self.len_x-self.len_y) < self.step else abs(self.len_x-self.len_y) # if a huge length difference
            path,distances = np.array([[0,0]]),np.array([self.metric(self.X[0].reshape((-1,1)),self.Y[0].reshape((-1,1)))[0][0]])
            
            while True:
                f = path[-1]
                t = (f[0]+self.step if f[0]+self.step < self.len_x else self.len_x,f[1]+self.step if f[1]+self.step < self.len_y else self.len_y)
                if t[0] >= self.len_x and t[1] >= self.len_y:
                    tmp_path,tmp_distances,_ = self._dtw(self.X[f[0]:t[0]],self.Y[f[1]:t[1]],self.metric)
                    path      = np.append(path,tmp_path[1:]+path[-1],axis=0)
                    distances = np.append(distances,tmp_distances[1:])
                    break
                elif t[0] >= self.len_x or t[1] >= self.len_y:
                    tmp_path,tmp_distances,_ = self._dtw(self.X[f[0]:t[0]],self.Y[f[1]:t[1]],self.metric)
                else:
                    tmp_path,tmp_distances,_ = self._dtw(self.X[f[0]:t[0]],self.Y[f[1]:t[1]],self.metric,early_stopping=True)
                path      = np.append(path,tmp_path[1:]+path[-1],axis=0)
                distances = np.append(distances,tmp_distances[1:])
                
            self.path,self.distances,self.total = path,distances,np.sum(distances)
            

class ddtw(dtw):
    """
    Derivative dtw. Loose implementation of the paper: https://www.ics.uci.edu/~pazzani/Publications/sdm01.pdf
    """
    def __init__(self,X,Y,metric=euclidean_distances,step=120):
        x, y = self._deriv(X), self._deriv(Y)
        dtw.__init__(self,x,y,metric,step)
        self.X, self.Y = X, Y
        self.path = np.r_[np.array([(0,0)]),self.path,np.array([(0,0)])]
        self.distances = np.r_[0,self.distances,0]
        self.len_x, self.len_y = len(X), len(Y)
    
    def _deriv(self,x):
        return (-np.diff(x)[:-1] - np.diff(x,2)/2)/2
    
    
if __name__ == 'main':
    import timeit
    
    n = 1000
    m = 50
    x = np.sin(np.linspace(0,2*np.pi,n+m))*3 + np.random.rand(n+m) + np.cos(np.linspace(0,5*np.pi,n+m))*2
    y = np.sin(np.linspace(0,2*np.pi,n))*3 + np.random.rand(n) + np.cos(np.linspace(0,5*np.pi,n))*1.5

   %timeit dtw(x,y)
    ww = dtw(x,y)
    w = ddtw(x,y)
    
    ww.plot()
    w.plot()
