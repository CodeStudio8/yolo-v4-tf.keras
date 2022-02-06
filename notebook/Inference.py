#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
tf.__version__


# In[2]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[3]:


import sys
sys.path.append("..")


# In[4]:


import cv2
import numpy as np
from glob import glob
from models import Yolov4


# In[5]:


model = Yolov4(weight_path='../yolov4.weights',
               class_name_path='../class_names/coco_classes.txt')


# In[7]:


model.predict('../img/street.jpeg', random_color=True)


# In[ ]:




