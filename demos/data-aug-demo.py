#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from scipy import ndimage
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[77]:


from google.colab import files
uploaded = files.upload()


# In[ ]:


imgnames = ['x_satImage_088.png']
x_train = np.array([np.asarray(cv2.imread(imgname)) for imgname in imgnames], dtype=np.float32)
imgnames = ['y_satImage_088.png']
y_train = np.array([np.asarray(cv2.imread(imgname)) for imgname in imgnames], dtype=np.float32)
imgx = x_train[0]
imgy = y_train[0]


# In[81]:


plt.subplot(121)
plt.imshow(imgx.astype(int))
plt.subplot(122)
plt.imshow(imgy.astype(int))
plt.show()


# In[89]:


smallx = cv2.resize(imgx, dsize=(263, 263), interpolation=cv2.INTER_CUBIC)
resizedx = cv2.resize(smallx, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
smally = cv2.resize(imgy, dsize=(263, 263), interpolation=cv2.INTER_CUBIC)
resizedy = cv2.resize(smally, dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
plt.subplot(121)
plt.imshow(resizedx.astype(int))
plt.subplot(122)
plt.imshow(resizedy.astype(int))
plt.show()


# In[82]:


rotx = ndimage.rotate(imgx, angle=90, order=1, reshape=False, axes=(0,1))
roty = ndimage.rotate(imgy, angle=90, order=1, reshape=False, axes=(0,1))
plt.subplot(121)
plt.imshow(rotx.astype(int))
plt.subplot(122)
plt.imshow(roty.astype(int))
plt.show()


# In[83]:


flipx = np.flipud(imgx)
flipy = np.flipud(imgy)
plt.subplot(121)
plt.imshow(flipx.astype(int))
plt.subplot(122)
plt.imshow(flipy.astype(int))
plt.show()


# In[84]:


crop_len = 100
zoomx = cv2.resize(imgx[crop_len:-crop_len,crop_len:-crop_len], dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
zoomy = cv2.resize(imgy[crop_len:-crop_len,crop_len:-crop_len], dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
plt.subplot(121)
plt.imshow(zoomx.astype(int))
plt.subplot(122)
plt.imshow(zoomy.astype(int))
plt.show()


# In[85]:


crop_len = 60
rotx_ = ndimage.rotate(imgx, angle=45, order=1, reshape=False, axes=(0,1))
zrotx = cv2.resize(rotx_[crop_len:-crop_len,crop_len:-crop_len], dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
roty_ = ndimage.rotate(imgy, angle=45, order=1, reshape=False, axes=(0,1))
zroty = cv2.resize(roty_[crop_len:-crop_len,crop_len:-crop_len], dsize=(400, 400), interpolation=cv2.INTER_CUBIC)
plt.subplot(121)
plt.imshow(zrotx.astype(int))
plt.subplot(122)
plt.imshow(zroty.astype(int))
plt.show()

