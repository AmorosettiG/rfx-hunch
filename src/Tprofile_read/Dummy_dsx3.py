from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import abc

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors 

import copy
import models.base

import Hunch_utils  as Htls



class ComposableAccess(Htls.Struct):
    __metaclass__ = abc.ABCMeta            
    _data = np.empty(1,dtype=None)

    def __init__(self, *ref):
        super().__init__(self, _data = ref[0])
        
    def __getattr__(self, name):
        return self._data[name]

    def __setattr__(self, name, val):
        self._data[name] = val
    
    def __getitem__(self, key):        
        return self.get_item_byname(key)

    def get_item_byname(self, key):
        fields = key.split('~')
        if len(fields) > 1: 
            val = np.concatenate([ np.atleast_1d(self.get_item_byname(k)) for k in fields ])
        else:
            try:    val = self._data[key]
            except: val = np.nan #np.full([len(self)], self._null)                
        return val   


class Dummy_dsx3(models.base.Dataset):


#        ____                               _                 
#   / ___|   __ _   _   _   ___   ___  (_)   __ _   _ __  
#  | |  _   / _` | | | | | / __| / __| | |  / _` | | '_ \ 
#  | |_| | | (_| | | |_| | \__ \ \__ \ | | | (_| | | | | |
#   \____|  \__,_|  \__,_| |___/ |___/ |_|  \__,_| |_| |_|
                                                        


    # kinds = [
    #     {'mean': [0.2,0.8], 'sigma': [0.1,0.1], 'gain': [1,1] },   # this is sum of 2 gaussion with fixed parameters   
    #     {'mean': [0.8], 'sigma': [0.1], 'gain': [0.5] },           # 1 gaussian with fixed parameters
    #     {'mean': [0.2], 'sigma': [0.1], 'gain': [0.5] },
    #     {'mean': [0.5], 'sigma': [0.2], 'gain': [1] },
    #     {'mean': [0.5], 'sigma': [0.2], 'gain': [0.5] },
    # ]

    # kinds = [   
        #  {'mean': [0.2], 'sigma': [0.2], 'gain': [0.3] },
        #  {'mean': [0.8], 'sigma': [0.2], 'gain': [0.3] },
        #  {'mean': [0.2], 'sigma': [0.2], 'gain': [0.9] },
        #  {'mean': [0.8], 'sigma': [0.2], 'gain': [0.9] },
        #  {'mean': [0.5], 'sigma': [0.2], 'gain': [0.5] },
        #  {'mean': [0.2,0.8], 'sigma': [0.1,0.1], 'gain': [1,1] },


        # {'mean': [(0.5,0.5)], 'sigma': [(0.2,0.2)], 'gain': [(0.1,0.9)] }, # only gain is variable    # 1 gaussian with variable parameters
        # {'mean': [(0.1,0.9)], 'sigma': [(0.2,0.2)], 'gain': [(0.5,0.5)] }, # only mean is variable
        # {'mean': [(0.5,0.5)], 'sigma': [(0.1,0.5)], 'gain': [(0.5,0.5)] }, # only sigma is variable
        # {'mean': [(0.2,0.8)], 'sigma': [(0.2,0.2)], 'gain': [(0.2,0.8)] }, # gain and mean variables
        # {'mean': [(0.2,0.8)], 'sigma': [(0.1,0.5)], 'gain': [(0.2,0.8)] }, # 3 parameters variables


        # {'mean': [(0.1,0.9),(0.1,0.9)], 'sigma': [(0.2,0.2),(0.2,0.2)], 'gain': [(0.1,0.9),(0.1,0.9)] },  # sum of 2 gaussians with variable parameters
        # {'mean': [(0.1,0.2),(0.4,0.5),(0.8,0.9)], 'sigma': [(0.1,0.3),(0.1,0.3),(0.1,0.3)], 'gain': [(0.1,0.3),(0.1,0.3),(0.1,0.3)] },
    # ]



#       ____    ____   __  __  _____ 
#  |  _ \  / ___|  \ \/ / |___ / 
#  | | | | \___ \   \  /    |_ \ 
#  | |_| |  ___) |  /  \   ___) |
#  |____/  |____/  /_/\_\ |____/ 
                               
    kinds = [

        ################ QSH 
        # (random values in ranges)
        {'A': [(1.4,4.)], 'B': [(6.,16.)], 'C': [(4.,12.)] },

        # (fixed values)
        # {'A': [1.8], 'B': [8.], 'C': [8.] },


        ################ MH 
        #(random values in ranges)
        {'A': [(1.,2.5)], 'B': [(3.,7.)], 'C': [(0.8,2.5)] },

        # (fixed values)
        # {'A': [1.4], 'B': [5.], 'C': [1.5] },

    ] 




    def __init__(self, counts=60000, size=20, noise_var=0., nanprob=None, nanmask=None, fixed_nanmask=None):
        self._counts = counts
        self._size = size
        self._noise = noise_var
        self._nanmask = None
        self._nanprob = None
        self._fixed_nanmask = None
        if nanmask is not None:
            self._nanmask = np.array(nanmask)
        if nanprob is not None:
            self._nanprob = np.array(nanprob)
        if fixed_nanmask is not None:
            self._fixed_nanmask = np.array(fixed_nanmask)
        self._null = np.nan
        self._dataset = None




    def buffer(self, counts = None):
        if counts is None: counts = self._counts
        else             : self._counts = counts
        size = self._size
        dtype = np.dtype ( [  ('x', '>f4', (size,) ),
                              ('y', '>f4', (size,) ),
                              ('y_min', np.float32),
                              ('y_max', np.float32),
                              ('y_mean', np.float32),
                              ('y_median', np.float32),
                              ('l_magic', np.float32),

                            #################################
                            #   ('l_mean_A', np.float32),
                            #   ('l_mean_B', np.float32),
                            #   ('l_mean_C', np.float32),
                            #################################

                            #   ('l_mean_gain', np.float32),
                            #   ('l_mean_sigma', np.float32),
                              ('l', np.int32),
                           ] )
        ds = np.empty([counts], dtype=dtype)
        for i in range(counts):
            s_pt,l = self.gen_pt(i)
            ds[i] = (
                     s_pt[:,0], s_pt[:,1], 
                     np.nanmin(s_pt[:,1]),
                     np.nanmax(s_pt[:,1]),
                     np.nanmean(s_pt[:,1]),
                     np.nanmedian(s_pt[:,1]),
                     float(l)/len(self.kinds),

                     #################################
                    #  np.mean(self.kinds[l]['A']),
                    #  np.mean(self.kinds[l]['B']),
                    #  np.mean(self.kinds[l]['C']),
                     #################################

                    #  np.mean(self.kinds[l]['gain']),
                    #  np.mean(self.kinds[l]['sigma']),
                     l, 
                    )
        self._dataset = ds
        return self
    
    def clear(self):
        self._dataset = None

    def __len__(self):
        return self._counts

    # return by reference
    def __getitem__(self, key):
        assert self._dataset is not None, 'please fill buffer first'
        if isinstance(key, int):
            return ComposableAccess(self._dataset[key])
            # return self._dataset[key]
        elif isinstance(key, range):
            return self._dataset[key]
        elif isinstance(key, slice):
            ds = copy.deepcopy(self)
            ds._dataset = self._dataset[key]
            ds._counts = len(ds._dataset)
            return ds
        elif isinstance(key, str):
            try:    val = self._dataset[:][key]
            except: val = np.full([self._counts], self._null)
            return val
        elif isinstance(key, tuple):
            val = [ self[:][k] for k in key ]
            return val
        else:
            print("not supported index: ",type(key))



    # set by reference
    def __setitem__(self, key, value):
        assert self._dataset is not None, 'please fill buffer first'
        if isinstance(key, int):
            self._dataset[key] = value
        elif isinstance(key, range) or isinstance(key, slice):
            self._dataset[key] = value
        elif isinstance(key, str):
            try: self._dataset[:][key] = value
            except: print('WARNING: field not found')
        else:
            print("not supported index: ",type(key))


    @property
    def dim(self):
        return self._size

    @property
    def size(self):
        return self._counts

    def gen_pt(self, id=None, x=None, kind=None):

        #### Gaussian
        def gauss(x, m, s, g):
            return np.abs(np.exp(-np.power(x-m, 2.) / (2 * np.power(s, 2.))) * g + np.random.normal(0,self._noise,1))

        def dsx3(x, a, b, c):
            return (1 - np.power(x,a)) + np.power((1 - np.power(x,b)),c)

        if self._dataset is not None and id is not None:
            data = self._dataset[id]
            return np.stack([data['x'],data['y']], axis=1), data['l']
        else:
            if x is None:
                # uniform array of x :
                x = np.linspace(0,1,self._size)
                #x = np.sort(np.random.rand(self._size)) # previous way : random array
            y = np.zeros_like(x)        
            if kind is None:
                kind = np.random.randint(len(self.kinds))
            k = self.kinds[kind]

            # print(list(k.keys())[0])


            if list(k.keys())[0] == 'mean':

                if type(k['mean'][0])==tuple:           
                    for i in range(len(k['mean'])):
                        m = k['mean'][i][0] + np.random.sample(1)[0]*(k['mean'][i][1]-k['mean'][i][0])
                        s = k['sigma'][i][0] + np.random.sample(1)[0]*(k['sigma'][i][1]-k['sigma'][i][0])
                        g = k['gain'][i][0] + np.random.sample(1)[0]*(k['gain'][i][1]-k['gain'][i][0])
                        y += gauss(x,m,s,g)
                else:
                    for i in range(len(k['mean'])):
                        m = k['mean'][i]
                        s = k['sigma'][i]
                        g = k['gain'][i]
                        y += gauss(x,m,s,g)

            
            
            elif list(k.keys())[0] == 'A':

                if type(k['A'][0])==tuple:
                    for i in range(len(k['A'])):
                        a = k['A'][i][0] + np.random.sample(1)[0]*(k['A'][i][1]-k['A'][i][0])
                        b = k['B'][i][0] + np.random.sample(1)[0]*(k['B'][i][1]-k['B'][i][0])
                        c = k['C'][i][0] + np.random.sample(1)[0]*(k['C'][i][1]-k['C'][i][0])
                        y += dsx3(x,a,b,c)
                else:
                    for i in range(len(k['A'])):
                        a = k['A'][i]
                        b = k['B'][i]
                        c = k['C'][i]
                        y += dsx3(x,a,b,c)

                


                # to remove :
                #if len(np.shape(k['mean'])) > 0 and 
                #for m,s,g in k['mean'],k['sigma'],k['gain']: # doesn't work
                # for m in k['mean'] :
                #     if isinstance(m,tuple): m = m[0] + np.random.sample(1)[0]*(m[1]-m[0])
                # for s in k['sigma'] :
                #     if isinstance(s,tuple): s = s[0] + np.random.sample(1)[0]*(s[1]-s[0])
                # for g in k['gain'] :
                #     if isinstance(g,tuple): g = g[0] + np.random.sample(1)[0]*(g[1]-g[0])
                #y = gauss(x,k['mean'],k['sigma'],k['gain'])

            
            mask = np.zeros_like(x)
            if self._nanprob is not None:
                mask = np.random.uniform(size=self._size)
                mask = (mask < self._nanprob).astype(float)

            # if self._nanmask is not None:
            #     mask = self._nanmask & np.random.randint(2, size=self._size)
            #     if self._fixed_nanmask is not None:
            #         mask = mask | self._fixed_nanmask
            
            x[mask > 0] = np.nan
            y[mask > 0] = np.nan
            return np.stack([x,y], axis=1), kind
    
    
    def get_tf_dataset_tuple(self):
        types = tf.float32, tf.float32, tf.int32
        shape = (self._size,),(self._size,),()
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    s_pt,l = self.gen_pt(i)
                    yield s_pt[:,0], s_pt[:,1], l
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)

    def get_tf_dataset_array(self):
        types = tf.float32, tf.int32
        shape = (2*self._size,),()
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    s_pt, l = self.gen_pt(i)
                    yield np.concatenate([s_pt[:,0], s_pt[:,1]]), l
                else:
                    return
        return tf.data.Dataset.from_generator(gen, types, shape)
        

    def tf_tuple_compose(self, fields=[]):
        def gen():
            import itertools
            for i in itertools.count(0):
                if i < len(self):
                    pt = self[i]
                    yield tuple([ pt[n] for n in fields])
                else:
                    return
        d0 = tuple([ self[0][n] for n in fields])
        types = tuple([tf.convert_to_tensor(x).dtype for x in d0])
        shape = tuple([np.shape(x) for x in d0])
        return tf.data.Dataset.from_generator(gen, types, shape)

    ds_tuple = property(get_tf_dataset_tuple)
    ds_array = property(get_tf_dataset_array)






def test_gendata(g1=None):
    if g1 is None:
        g1 = Dummy_dsx3()
    x,y,_ = g1.ds_tuple.batch(200).make_one_shot_iterator().get_next()
    plt.figure('g1')
    plt.clf()
    for x,y in np.stack([x,y],axis=1):
        plt.plot(x,y,'.')