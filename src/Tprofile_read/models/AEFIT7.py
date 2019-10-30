
# Model for testing the KL loss in custom layer (embed sthocastic reparametrization in single layer)
# this should provide a model that can be compiled in a single net.

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



##
## LAYER for adding Variational on Latent variables
##
class VAE_latent(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(VAE_latent, self).__init__( *args, **kwargs )

    def call(self, x):

        def reparametrize(z_mean, z_log_var):
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        x = tf.convert_to_tensor(x)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        kl_loss = -0.5 * tf.reduce_sum(1. + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
        mean = reparametrize(mean, logvar)        
        self.add_loss(kl_loss)
        return mean, logvar





"""
.##.....##..#######..########..########.##......
.###...###.##.....##.##.....##.##.......##......
.####.####.##.....##.##.....##.##.......##......
.##.###.##.##.....##.##.....##.######...##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##.##.....##.##.....##.##.......##......
.##.....##..#######..########..########.########
"""
class AEFIT7(models.AEFIT5.AEFIT5):
    ''' General Autoencoder Fit Model for TF 2.0
    '''    
    def __init__(self, feature_dim=40, latent_dim=2, dprate=0., activation=tf.nn.relu, beta=1., 
                 geometry=[20,20,10,10], scale=1, *args, **kwargs):
        super().__init__(feature_dim, latent_dim, dprate, activation, beta, geometry, scale, *args, **kwargs)
        self.compile(
            loss=self.compute_mse_loss
            )
        print('AEFIT7 ready:')
    


    def set_model(self, feature_dim, latent_dim, dprate=0., activation=tf.nn.relu, 
                  geometry=[20,20,10], scale=1):

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
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            if len(geometry) == 0: initializer = LsInitializer()
            else : initializer = None            
            self.add(tf.keras.layers.Dense(ldim, activation='linear', use_bias=False, kernel_initializer=initializer))
            return self

        def add_dense_decode(self, fdim=feature_dim, ldim=latent_dim, geometry=[10,10,20,20]):            
            for _,size in enumerate(geometry):
                self.add(tf.keras.layers.Dense(fdim*size*scale, activation=activation))
                self.add(tf.keras.layers.Dropout(dprate))
            if len(geometry) == 0: initializer = tf.initializers.identity()
            else : initializer = None
            self.add(tf.keras.layers.Dense(fdim, activation='linear', use_bias=False, kernel_initializer=initializer))            
            return self
        # add methods to Sequential class
        tf.keras.Sequential.add_dense_encode = add_dense_encode
        tf.keras.Sequential.add_dense_decode = add_dense_decode
        
        ## INFERENCE ##
        inference_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(feature_dim,)),
            models.layers.NaNDense(feature_dim),
            models.layers.Relevance1D(name=self.name+'_iRlv', activation='linear', kernel_initializer=tf.keras.initializers.ones),
        ]).add_dense_encode(ldim=2*latent_dim, geometry=geometry)                
        inference_net.add(VAE_latent(latent_dim))

        ## GENERATION ##
        generative_net = tf.keras.Sequential( [
            tf.keras.layers.Input(shape=(latent_dim,)),
            models.layers.Relevance1D(name=self.name+'_gRlv', activation='linear', kernel_initializer=tf.keras.initializers.ones),
        ]).add_dense_decode(geometry=geometry[::-1])
        
        self.inference_net = inference_net
        self.generative_net = generative_net
        
        return inference_net, generative_net

    
    def encode(self, X, training=None):
        return self.inference_net(X, training)        
    
    def decode(self, s, training=True, apply_sigmoid=None):
        x = self.generative_net(s, training=training)
        if apply_sigmoid is None: apply_sigmoid = self.apply_sigmoid        
        if apply_sigmoid is True and training is False:
            x = tf.sigmoid(x)
        return x

    def call(self, xy, training=True):
        att = tf.math.is_nan(xy)
        xy  = tf.where(att, tf.zeros_like(xy), xy)
        z,_ = self.encode(xy, training=training)
        XY  = self.decode(z, training=training)        
        if training is not False:
            XY  = tf.where(att, tf.zeros_like(XY), XY)
        return XY
 
    def compile(self, optimizer=None, loss=None, logit_loss=False, metrics=None, **kwargs):
        return super().compile(optimizer, loss, logit_loss, metrics, **kwargs)

    def compute_mse_loss(self, xy, XY):
        return tf.keras.losses.mse(y_pred=XY, y_true=xy) + self.losses[0]








