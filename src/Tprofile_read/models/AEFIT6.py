


                # self.add(tf.keras.layers.Dense(fdim*size*scale, use_bias=False))
                # self.add(tf.keras.layers.BatchNormalization())
                # self.add(tf.keras.layers.Activation(activation))






from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import preprocessing

import tensorflow as tf
import abc

# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import ipysh
import models
import models.layers

"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""
class AEFIT6(models.AEFIT5.AEFIT5):
    ''' General Autoencoder Fit Model for TF 2.0
    '''    
    def __init__(self, feature_dim=40, latent_dim=2, dprate=0., activation=tf.nn.relu, beta=1., 
                 geometry=[20,20,10,10], scale=1, *args, **kwargs):
        super().__init__(feature_dim, latent_dim, dprate, activation, beta, geometry, scale, *args, **kwargs)
        print('AEFIT6 ready:')
    
    def set_model(self, feature_dim, latent_dim, dprate=0., activation=tf.nn.relu, 
                  geometry=[20,20,10,10], scale=1):

        class LsInitializer(tf.keras.initializers.Initializer):
            """Initializer for latent layer"""
            def __init__(self, axis=1):
                super(LsInitializer, self).__init__()
                self.axis = axis

            def __call__(self, shape, dtype=tf.dtypes.float32):
                dtype = tf.dtypes.as_dtype(dtype)
                if not dtype.is_numpy_compatible or dtype == tf.dtypes.string:
                    raise ValueError("Expected numeric or boolean dtype, got %s." % dtype)
                axis = self.axis
                shape[axis] = int(shape[axis]/2)
                identity = tf.initializers.identity()(shape)
                return tf.concat([identity, tf.zeros(shape)], axis=axis)

        def add_dense_encode(self, fdim=feature_dim, ldim=latent_dim, geometry=[20,20,10,10]):
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale))
                #self.add(tf.keras.layers.BatchNormalization(  center=False, scale=False ))
                self.add(tf.keras.layers.Activation(activation))
            if len(geometry) == 0: initializer = LsInitializer()
            else : initializer = None
            self.add(tf.keras.layers.Dense(ldim, activation='linear', kernel_initializer=initializer))
            return self

        def add_dense_decode(self, fdim=feature_dim, ldim=latent_dim, geometry=[10,10,20,20]):            
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, use_bias=False))
                self.add(tf.keras.layers.BatchNormalization( center=False, scale=False))
                self.add(tf.keras.layers.Activation(activation))
            if len(geometry) == 0: initializer = tf.initializers.identity()
            else : initializer = None
            self.add(tf.keras.layers.Dense(fdim, activation='linear', kernel_initializer=initializer))            
            return self
        # add methods to Sequential class
        tf.keras.Sequential.add_dense_encode = add_dense_encode
        tf.keras.Sequential.add_dense_decode = add_dense_decode
        
        ## INFERENCE ##
        inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            tf.keras.layers.Lambda(lambda x: tf.where(tf.math.is_nan(x),tf.zeros_like(x),x)), 
            # # NaNDense(feature_dim),
            models.layers.Relevance1D(name=self.name+'_iRlv', activation='linear', kernel_initializer=tf.initializers.ones),
        ]).add_dense_encode(ldim=2*latent_dim, geometry=geometry)

        ## GENERATION ##
        generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            models.layers.Relevance1D(name=self.name+'_gRlv', activation='linear', kernel_initializer=tf.initializers.ones),
        ]).add_dense_decode(geometry=geometry[::-1])
        
        self.inference_net = inference_net
        self.generative_net = generative_net        
        return inference_net, generative_net

 
    def compile(self, optimizer=None, loss=None, logit_loss=False, metrics=None, **kwargs):
        return super().compile(optimizer, loss, logit_loss, metrics, **kwargs)

    