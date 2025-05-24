import torch
import torch.nn as nn
import torch.nn.functional as F
from stutils.STmodel import *
import pandas as pd
from collections import defaultdict
import numpy as np
from einops.layers.torch import Rearrange

class geneModel(nn.Module):

    def __init__(self,params,st_tokenizer):
        super().__init__()

        hidden_dim = params['hidden_dim']
        n_head = params['n_head']
        ff_dim = params['ff_dim']
        n_layers = params['n_layers']
        self.property_width = params['property_width']
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=params['property_width'])
        self.hidden_dim = hidden_dim
        ST_MODEL_DEFAULT_SETTINGS = {
        "max_len": 128,  
        "pp_v_dim": 7 + 1,  
        "pp_e_dim": 1, 
        "pp_encoder_n_layer": 4,  
        "hidden_dim": 384,  
        "n_layers": 8, 
        "ff_dim": 1024,  
        "n_head": 8,  
        "remove_pp_dis": False, 
        "non_vae": False, 
        "prop_dim":256,
        "property_width":157,
        "temperature":1.0,
        }
        model_params = dict(ST_MODEL_DEFAULT_SETTINGS)
        smiles_model = STransformer(model_params, st_tokenizer)

        self.smiencoder = smiles_model
        self.attention = MultiheadAttention(hidden_dim, n_head, dropout=0.1)
        self.pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))
        self.zz_seg_encoding = nn.Parameter(torch.randn(hidden_dim))
        self.property_mask = nn.Parameter(torch.zeros(1, 1, params['hidden_dim']))   
        self.property_embed = nn.Linear(1, hidden_dim)
        self.propencoder = TransformerEncoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)

        self.expand = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                Rearrange('batch_size h -> 1 batch_size h')
            )

        dfdd = pd.read_csv('./datasets/example.csv')
        header_list = dfdd.columns.tolist()
        header_dict = {header: idx-1 for idx, header in enumerate(header_list)}
        self.header_dict = header_dict

        df = pd.read_csv('./datasets/pair.csv')
        feature_dict = defaultdict(list)
        for _, row in df.iterrows():
            feature_dict[header_dict[row['Feature1']]].append(header_dict[row['Feature2']])
        self.feature_dict = feature_dict
        
        self.dencoder = TransformerEncoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers) 
        self.decoder = TransformerDecoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)


    @torch.jit.ignore
    def process_p_n(self, prop,value = 0.3,sm = True):

        property_feature = self.property_embed(prop.unsqueeze(2))  
        batch_size, num_properties, feature_dim = property_feature.shape
        mpm_mask = torch.bernoulli(torch.ones(batch_size,num_properties) * value) .to(property_feature.device)  
        mpm_mask_copy = mpm_mask.clone()
        if sm:
            for key, value in self.feature_dict.items():
                mask = (mpm_mask_copy[:, key] == 1)  
                if mask.any():  
                    true_indices = mask.nonzero(as_tuple=True)[0]
                    mpm_mask[true_indices[:, None], value]=1

        pmask = self.property_mask.expand(batch_size, num_properties, feature_dim).to(property_feature.device)
        properties = torch.where(mpm_mask.unsqueeze(-1) == 1, pmask, property_feature).to(property_feature.device)

        xt = properties.permute(1, 0, 2).contiguous()  # (seq,batch,feat)
        xt = self.pos_encoding(xt)
        xxt = self.propencoder(xt, mpm_mask)  # (s b f), input masks need not transpose
        xxt = xxt + self.pp_seg_encoding

        foo = xxt.new_ones(1, *xxt.shape[1:])
        z, _ = self.attention(foo, xxt, xxt, key_padding_mask=mpm_mask)
        z = z.squeeze(0)

        return z,mpm_mask,xxt

    
    def expand_then_fusing(self, z, pp_mask, vvs):
        zz = self.expand(z.squeeze(0))  
        zz = self.smiencoder.pos_encoding(zz)  
        zzs = zz + self.zz_seg_encoding 

        # cat pp and latent
        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        full_mask = torch.cat((pp_mask, full_mask), dim=1)  

        zzz = torch.cat((vvs, zzs), dim=0)  
        zzz = self.dencoder(zzz, full_mask) 

        return zzz, full_mask

       