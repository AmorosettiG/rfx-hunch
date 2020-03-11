
import torch

from brevitas.core.quant import QuantType
from brevitas.core.quant import RescalingIntQuant
from torch.nn import Module, ModuleList, BatchNorm1d, Dropout, Linear, Sequential
from .common import get_quant_linear, get_act_quant, get_quant_type, get_stats_op

INTERMEDIATE_FC_PER_OUT_CH_SCALING = True
LAST_FC_PER_OUT_CH_SCALING = False
IN_DROPOUT = 0.2
HIDDEN_DROPOUT = 0.2
FC_OUT_FEATURES = [32,32,32]

weight_quant_type = get_quant_type(1)
act_quant_type = get_quant_type(1)
in_quant_type = get_quant_type(1)
stats_op = get_stats_op(weight_quant_type)

class VAE(Module):
    def __init__(self, feature_dim=30, latent_dim=16, size=1, geometry=FC_OUT_FEATURES):
        
        fcl = lambda in_dim, out_dim : get_quant_linear(in_dim, out_dim,
                                                        per_out_ch_scaling=INTERMEDIATE_FC_PER_OUT_CH_SCALING,
                                                        bit_width=1, quant_type=weight_quant_type,
                                                        stats_op=stats_op)
        fcl_last = lambda in_dim, out_dim : get_quant_linear(in_dim, out_dim,
                                                        per_out_ch_scaling=LAST_FC_PER_OUT_CH_SCALING,
                                                        bit_width=1, quant_type=weight_quant_type,
                                                        stats_op=stats_op)        
        act = lambda: get_act_quant(act_bit_width=1, act_quant_type=act_quant_type)            
        fcs = lambda out_features : int(out_features*size)
        
        super(VAE, self).__init__()
        self.feature_dim = feature_dim
        
        self.fc1 = Sequential()
        # self.fc1.add_module('fc1_act', act())
        # self.fc1.append(Dropout(p=IN_DROPOUT))
        in_features = feature_dim
        for i,out_features in enumerate(geometry):
            out_features = fcs(out_features)
            self.fc1.add_module('fc1_'+str(i), fcl(in_features, out_features))
            self.fc1.add_module('fc1_'+str(i)+'bn', BatchNorm1d(num_features=out_features))
            self.fc1.add_module('fc1_'+str(i)+'act', act())     
            self.fc1.add_module('fc1_'+str(i)+'dp', Dropout(p=HIDDEN_DROPOUT))
            in_features = out_features
        self.fc1.add_module('fc1_last', fcl_last(in_features, in_features))  
        
        self.fc21 = Linear(in_features, latent_dim)
        self.fc22 = Linear(in_features, latent_dim)

        self.fc3 = Sequential()
        self.fc3.add_module('fc3_act', act())
        # self.fc1.append(Dropout(p=IN_DROPOUT))
        in_features = latent_dim
        for i,out_features in enumerate(geometry[::-1]):
            out_features = fcs(out_features)
            self.fc3.add_module('fc3_'+str(i), fcl(in_features, out_features))
            self.fc3.add_module('fc3_'+str(i)+'bn', BatchNorm1d(num_features=out_features))
            self.fc3.add_module('fc3_'+str(i)+'act', act())
            self.fc3.add_module('fc3_'+str(i)+'dp', Dropout(p=HIDDEN_DROPOUT))
            in_features = out_features
        self.fc3.add_module('fc3_last', fcl_last(in_features, in_features))        
        self.fc4 = Linear(in_features, feature_dim)

    def get_params(self):
        return self.mu, self.logvar
        
    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = self.fc3(z)
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        Module.eval
        self.mu, self.logvar = self.encode(x.view(-1,  self.feature_dim))
        if self.training: z = self.reparameterize(self.mu, self.logvar)
        else: z = self.mu
        return self.decode(z)
    
    def get_inference_net(self):
        class inference_net(Module):
            def __init__(self, parent):
                super(inference_net, self).__init__()
                self.fc1 = parent.fc1
                self.fc21 = parent.fc21                

            def forward(self, x):
                return self.fc21(self.fc1(x))
        return inference_net(self)



