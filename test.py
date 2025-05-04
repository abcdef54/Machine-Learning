import numpy as np
import torch

a = []
a = np.array(a)

print(type(a) if isinstance(a, (list, np.ndarray)) else 'Tensor')