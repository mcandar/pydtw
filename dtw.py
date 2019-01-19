import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class dtw:
    """
    Dynamic time warping calculation.
    """
    def __init__(self,X,Y,metric=euclidean_distances,delta=120):
        self.X = X
        self.Y = Y
        self.metric = metric
        len_x, len_y = len(X), len(Y)
        if delta is None:
            self.path,self.distances,self.total = self._dtw(X,Y,metric)
        else:
            delta = delta if abs(len_x-len_y) < delta else abs(len_x-len_y) # if a huge length difference
            path,distances,total = np.array([[0,0]]),np.array([0]),0
            it = max((len_x//delta)+1,(len_y//delta)+1)
            for i in range(it):
                f = path[-1]
                t = (f[0]+delta if f[0]+delta < len_x else len_x,f[1]+delta if f[1]+delta < len_y else len_y)
                if i == it-1:
                    tmp_path,tmp_distances,_ = _dtw(X[f[0]:t[0]],Y[f[1]:t[1]],metric)
                else:
                    tmp_path,tmp_distances,_ = _dtw(X[f[0]:t[0]],Y[f[1]:t[1]],metric,early_stopping=True)
                path      = np.append(path,tmp_path[1:]+path[-1],axis=0)
                distances = np.append(distances,tmp_distances[1:])
            
            self.path,self.distances,self.total = path,distances,np.sum(distances)
            
    def _dtw(self,X,Y,metric=euclidean_distances):
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
                distance = [self.cost_matrix[I,J+1]] # to match length
                direction = 0 # compulsory direction
            elif J >= self._len_y and I < self._len_x:
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
        x = np.arange(self._len_x+1)
        y = np.arange(self._len_y+1)
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
