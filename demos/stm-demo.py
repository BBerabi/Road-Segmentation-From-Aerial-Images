#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys 
sys.path.append('..')
from submission_to_mask import reconstruct_from_labels
import matplotlib.pyplot as plt
import numpy as np


# In[9]:


im = reconstruct_from_labels('./submission_resize.csv', 7)
plt.imshow(im.astype(int), cmap='gray')
plt.show()

