import numpy as np
from typing import Union

class BaseCycleIndex:
    #Base class to cycle through indices in batches with optional shuffling.
    def __init__(self, indices: Union[int, list], batch_size: int, shuffle: bool = True) -> None:
        if isinstance(indices, int):
            indices = np.arange(indices)
        self.indices = np.array(indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pointer = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def _next_indices(self, arr, pointer, size):
        #Get size elements from arr, cycling if needed.
        end = pointer + size
        if end <= len(arr):
            out = arr[pointer:end]
            pointer = end % len(arr) # if end == len(arr) return 0 set pointer to zero otherwise to len(arr)
            end_reached = False 
        else:
            out = np.concatenate((arr[pointer:], arr[:end % len(arr)]))
            pointer = end % len(arr)
            end_reached = True
        return out, end_reached, pointer 
            
    def get_batch_ind(self):
        #Get next batch indices.
        batch, end_reached, self.pointer = self._next_indices(self.indices, self.pointer, self.batch_size)
        if end_reached and self.shuffle:
            np.random.shuffle(self.indices)
        return batch

    
class CycleIndex(BaseCycleIndex):
    #CycleIndex just inherits base behavior — no change needed.

    pass


class CycleIndexBalanced(BaseCycleIndex):
    #Class to generate class-balanced batches of training ids, with at least m samples per batch, shuffled after each epoch.
    
    def __init__(self, indices: Union[int,list], 
                y: Union[int,list],
                batch_size: int,
                min_class1: int = 1,
                shuffle: bool = True) -> None:
        if isinstance(indices, int):
            indices = np.arange(indices)
        self.indices = np.array(indices)
        self.y = np.array(y)
        self.batch_size = batch_size
        self.min_class1 = min_class1
        self.shuffle = shuffle

        # separate indices by class
        self.class1_inds = self.indices[self.y == 1]
        self.class0_inds = self.indices[self.y == 0]

        self.pointer0 = 0
        self.pointer1 = 0

        if self.shuffle:
            np.random.shuffle(self.class1_inds)
            np.random.shuffle(self.class0_inds)

    
    def get_batch_ind(self):
        # Get next balanced batch of indices
        # Get class 1 samples
        n1 = min(self.min_class1, len(self.class1_inds))
        class1_batch, end1_reached, self.pointer1 = self._next_indices(self.class1_inds, self.pointer1, n1)

        # Get remaining samples from class 0
        n0 = self.batch_size - len(class1_batch)
        class0_batch, end0_reached, self.pointer0 = self._next_indices(self.class0_inds, self.pointer0, n0)

        if self.shuffle and end1_reached:
            np.random.shuffle(self.class1_inds)

        if self.shuffle and end0_reached:
            np.random.shuffle(self.class0_inds)

        batch = np.concatenate((class1_batch, class0_batch))
        np.random.shuffle(batch)
        return batch