import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

class Dtw:
    """
    Dynamic time warping calculation.
    """
    def __init__(self,*args,**kwargs):
        result = self._dtw(*args,**kwargs)
        self.path = result['path']
        self.distances = result['distances']
        self.total = result['total']
    
    def _dtw(self,X,Y,metric=euclidean_distances):
        len_x,len_y = X.shape[0]-1, Y.shape[0]-1
        I, J = 0, 0
        path = [(0,0)]
        if len(X.shape) == 1 or len(Y.shape) == 1:
            dist_mat = metric(X.reshape((-1,1)),Y.reshape((-1,1)))
        else:
            dist_mat = metric(X,Y)
        distances = [dist_mat[I,J]]
        
        for n in range(len_x+len_y): 
            # check boundary conditions
            if I >= len_x and J < len_y:
                distance = [dist_mat[I,J+1]] # to match length
                direction = 0 # compulsory direction
            elif J >= len_y and I < len_x:
                distance = [-1,dist_mat[I+1,J]] # to match length
                direction = 1 # compulsory direction
            elif I >= len_x and J >= len_y:
                break
            else:
                distance = [dist_mat[I,J+1], # to right
                            dist_mat[I+1,J], # to above
                            dist_mat[I+1,J+1]] # to right and above
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
            
        return dict(path = path, distances = distances,total = np.sum(distances))