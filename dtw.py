#%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class dtw:
    """
    Dynamic time warping calculation.
    """
    def __init__(self,X,Y,metric=euclidean_distances):
        self.X = X
        self.Y = Y
        self.metric = metric
        self.path,self.distances,self.total = self._dtw(X,Y,metric)
    
    def _dtw(self,X,Y,metric=euclidean_distances):
        self._len_x,self._len_y = X.shape[0]-1, Y.shape[0]-1
        I, J = 0, 0
        path = [(0,0)]
        if len(X.shape) == 1 or len(Y.shape) == 1:
            self.cost_matrix = metric(X.reshape((-1,1)),Y.reshape((-1,1)))
        else:
            self.cost_matrix = metric(X,Y)
        distances = [self.cost_matrix[I,J]]
        
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
        return path,distances,np.sum(distances)
    
    def plot(self):
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
        plt.show(ax)
