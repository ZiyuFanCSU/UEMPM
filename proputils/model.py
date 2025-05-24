import torch
import torch.nn as nn
import torch
import torch.nn as nn
from stutils.STmodel import *

class PropModel(nn.Module):

    def __init__(self,params ,smiles_model):
        super().__init__()
        hidden_dim = params['hidden_dim']
        self.property_width = params['property_width']
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=params['property_width'])
        self.hidden_dim = hidden_dim
        self.smiencoder = smiles_model

    @torch.jit.unused
    def forward_embedding(self, inputs, input_mask):
        x = self.smiencoder.word_embed(inputs)
        xt = x.permute(1, 0, 2).contiguous()  
        xt = self.smiencoder.pos_encoding(xt)
        xxt = self.smiencoder.encoder(xt, input_mask) 
        foo = xxt.new_ones(1, *xxt.shape[1:])
        z, _ = self.smiencoder.attention(foo, xxt, xxt, key_padding_mask=input_mask)

        return z

class MModel(nn.Module):

    def __init__(self,st_tokenizer,fc_in_features, num_tasks):
        super().__init__()
        
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
        phargnn_model = PropModel(model_params,smiles_model)
        state_dict = torch.load("./pretrained_models/predict.pth") 
        new_state_dict = {}
        for key in state_dict:
            new_key = key.replace('module.', '')  
            new_state_dict[new_key] = state_dict[key]

        phargnn_model.load_state_dict(new_state_dict, strict=False)

        self.model_pre = phargnn_model
        hidden_dim = 128
        self.fc = nn.Sequential(
            nn.Linear(fc_in_features, hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),  
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_tasks)  
        )

    def forward(self, inputs, input_mask):
        zzz = self.model_pre.forward_embedding(inputs, input_mask)
        result = self.fc(zzz.squeeze())

        return result
    
