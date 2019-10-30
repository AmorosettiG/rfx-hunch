# coding: utf-8

# In[25]:


import numpy as np
import tensorflow as tf

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import Hunch_utils  as Htls
import Hunch_lsplot as Hplt
import Hunch_tSNEplot as Hsne


get_ipython().magic('aimport Dataset_QSH')


# In[26]:

import torch
import torch.nn.functional as F
import brevitas.nn as qnn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


# In[59]:


from brevitas.core.quant import QuantType
from brevitas.core.quant import RescalingIntQuant
# INT-8
class TestModel(torch.nn.Module):
    def __init__(self):

        super(TestModel, self).__init__()
        self.fc1   = qnn.QuantLinear(30, 120, bias=True, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)

        self.relu3 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)

        self.relu4 = qnn.QuantReLU(quant_type=QuantType.INT, bit_width=8, max_val=6)
        self.fc3   = qnn.QuantLinear(84, 1, bias=False, 
                                     weight_quant_type=QuantType.INT, 
                                     weight_bit_width=8)

    def forward(self, x):
        out = self.relu3(self.fc1(x))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out


# In[65]:


qsh = Dataset_QSH.Dataset_QSH()
file = ipysh.abs_builddir+'/te_db_r15_clean_shuffle.npy'
qsh.load(file)

qsh.dim = 15
qsh.set_null(np.nan)
qsh.set_normal_positive()


# In[67]:


params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
ds = qsh.get_torch_dataset(**params)


# In[70]:


m = TestModel()
x = torch.randn(1, 30, requires_grad=True)
X = m(x)


# In[71]:


# m.eval()
batch_size = 1
# Export the model
torch.onnx.export(m,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "test_model.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})

