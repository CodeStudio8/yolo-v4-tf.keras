#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("..")


# In[2]:


from utils import DataGenerator, read_annotation_lines
from models import Yolov4
from config import yolo_config


# In[ ]:


train_lines, val_lines = read_annotation_lines('../dataset/txt/anno-test.txt', test_size=0.1)
FOLDER_PATH = '../dataset/img'
class_name_path = '../class_names/bccd_classes.txt'
data_gen_train = DataGenerator(train_lines, class_name_path, FOLDER_PATH)
data_gen_val = DataGenerator(val_lines, class_name_path, FOLDER_PATH)

model = Yolov4(weight_path=None, 
               class_name_path=class_name_path)

model.fit(data_gen_train, 
          initial_epoch=0,
          epochs=10000, 
          val_data_gen=data_gen_val,
          callbacks=[])


# In[ ]:





# In[ ]:




