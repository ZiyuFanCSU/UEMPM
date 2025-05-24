import torch
import torch.nn as nn
import torch.nn.functional as F
from stutils.STmodel import *
import pandas as pd
from collections import defaultdict
import numpy as np
from einops.layers.torch import Rearrange

class PropModel(nn.Module):

    def __init__(self,params ,smiles_model):
        super().__init__()

        hidden_dim = params['hidden_dim']
        n_head = params['n_head']
        ff_dim = params['ff_dim']
        n_layers = params['n_layers']
        self.property_width = params['property_width']
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=params['property_width'])
        self.hidden_dim = hidden_dim

        self.smiencoder = smiles_model
        self.pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))
        self.zz_seg_encoding = nn.Parameter(torch.randn(hidden_dim))
        self.property_mask = nn.Parameter(torch.zeros(1, 1, params['hidden_dim']))   
        self.property_embed = nn.Linear(1, hidden_dim)
        self.propencoder = TransformerEncoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)
        self.attention = MultiheadAttention(hidden_dim, n_head, dropout=0.1)
        self.attention_S = MultiheadAttention(hidden_dim, n_head, dropout=0.1)
        self.attention_P = MultiheadAttention(hidden_dim, n_head, dropout=0.1)
        self.mapping_s2g = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 128)
        )

        self.mapping_g2s = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 128)
        )
        self.mapping_s2g2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 128)
        )

        self.mapping_g2s2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 128)
        )

        self.expand = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                Rearrange('batch_size h -> 1 batch_size h')
            )
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        

        self.temperature = params['temperature']

        self.logit_scale = nn.Parameter(torch.ones([1]))
        self.logit_scale2 = nn.Parameter(torch.ones([1]))
        self.logit_scale3 = nn.Parameter(torch.ones([1]))
        
        self.dencoder = TransformerEncoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers) 
        self.decoder = TransformerDecoder(hidden_dim, ff_dim, num_head=n_head, num_layer=n_layers)

        dfdd = pd.read_csv('./datasets/example.csv')
        header_list = dfdd.columns.tolist()
        header_dict = {header: idx-1 for idx, header in enumerate(header_list)}
        self.header_dict = header_dict

        df = pd.read_csv('./datasets/pair.csv')
        feature_dict = defaultdict(list)
        for _, row in df.iterrows():
            feature_dict[header_dict[row['Feature1']]].append(header_dict[row['Feature2']])
        self.feature_dict = feature_dict

        self.zzs_fuse_linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.vvs_fuse_linear = nn.Linear(2 * hidden_dim, hidden_dim)

    @torch.jit.ignore
    def process_p(self, prop,value = 0.3,sm = True):
        if value != 0:
            values    =     [0.2, 0.3, 0.5, 0.4]  
            probabilities = [0.2, 0.3, 0.2, 0.3]
            value = np.random.choice(values, p=probabilities)

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

        xt = properties.permute(1, 0, 2).contiguous()  
        xt = self.pos_encoding(xt)
        xxt = self.propencoder(xt, mpm_mask) 
        xxt = xxt + self.pp_seg_encoding

        foo = xxt.new_ones(1, *xxt.shape[1:])
        z, _ = self.attention(foo, xxt, xxt, key_padding_mask=mpm_mask)
        z = z.squeeze(0)

        return z,mpm_mask,xxt

    
    def expand_then_fusing(self, z, pp_mask, vvs_orig):
        zz = self.expand(z.squeeze(0))  
        zz = self.smiencoder.pos_encoding(zz)  
        zzs_orig = zz + self.zz_seg_encoding 

        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        zzs_attn = self.attention_S(zzs_orig, vvs_orig, vvs_orig, key_padding_mask=pp_mask)[0]
        vvs_attn = self.attention_P(vvs_orig, zzs_orig, zzs_orig, key_padding_mask=full_mask)[0]

        zzs = torch.cat([zzs_orig, zzs_attn], dim=-1) 
        zzs = self.zzs_fuse_linear(zzs)

        vvs = torch.cat([vvs_orig, vvs_attn], dim=-1) 
        vvs = self.vvs_fuse_linear(vvs)

        full_mask = torch.cat((pp_mask, full_mask), dim=1)  
        zzz = torch.cat((vvs, zzs), dim=0) 
        zzz = self.dencoder(zzz, full_mask) 

        return zzz, full_mask

    @torch.jit.unused
    def forward(self,fi,ft,inputs, input_mask, targets,target_mask, props,batch_size,
                 smi_kl = True,return_dict = True,if_gene = False,sm = True):
        if props != None:
            batch_size = inputs.shape[0]
            if smi_kl:
                z, kl_loss_z = self.smiencoder.calculate_z(inputs, input_mask)

            _, target_length = targets.shape
            target_maskk = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool),
                                    diagonal=1).to(inputs.device)
            target_embed = self.smiencoder.word_embed(targets)
            target_embed = self.smiencoder.pos_encoding(target_embed.permute(1, 0, 2).contiguous())

            output = self.smiencoder.decoder(target_embed, z,
                                x_mask=target_maskk, mem_padding_mask=z.new_zeros(z.shape[1], z.shape[0])).permute(1, 0, 2).contiguous()
            prediction_scores = self.smiencoder.word_pred(output)  

            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_loss = F.cross_entropy(shifted_prediction_scores.view(-1, self.smiencoder.vocab_size), targets[:, 1:].contiguous().view(-1),
                                    ignore_index=self.smiencoder.pad_value)

            x = self.smiencoder.word_embed(inputs)
            xt = x.permute(1, 0, 2).contiguous() 
            xt = self.smiencoder.pos_encoding(xt)
            zz = self.smiencoder.encoder(xt, input_mask)
            zz = self.mapping_s2g(zz)

            x2 = self.smiencoder.word_embed(targets)
            xt2 = x2.permute(1, 0, 2).contiguous() 
            xt2 = self.smiencoder.pos_encoding(xt2)
            z2 = self.smiencoder.encoder(xt2, target_mask)
            zz2 = self.mapping_s2g(z2)

            v,vm,vvs = self.process_p(props,value=0,sm=sm)
            v2,vm2,vvs2 = self.process_p(props,sm=sm)
            while torch.equal(vm, vm2):
                v2,vm2 = self.process_p(props,sm=sm)

            v_c = self.mapping_g2s(vvs)
            v2_c = self.mapping_g2s(vvs2)
            v_c_v = self.mapping_g2s2(v)
            v2_c_v = self.mapping_g2s2(v2)

            logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
            logit_scale2 = torch.clamp(self.logit_scale2.exp(), max=100)
            logit_scale3 = torch.clamp(self.logit_scale3.exp(), max=100)

            phar_features_1_all = (v_c_v / (v_c_v.norm(dim=-1, keepdim=True)+1e-10))
            phar_features_2_all = (v2_c_v / (v2_c_v.norm(dim=-1, keepdim=True)+1e-10))
            smiles_features_all = (zz / (zz.norm(dim=-1, keepdim=True)+1e-10)).permute(1, 0, 2)[:,0]
            smiles_features_aug_all = (zz2 / (zz2.norm(dim=-1, keepdim=True)+1e-10)).permute(1, 0, 2)[:,0]
            
            a_logits_per_phar_1 = (logit_scale * phar_features_1_all  @ smiles_features_all.T).view(batch_size,batch_size)
            a_logits_per_phar_2 = (logit_scale * phar_features_2_all  @ smiles_features_all.T).view(batch_size,batch_size)
            a_logits_per_phar_1_aug = (logit_scale * phar_features_1_all  @ smiles_features_aug_all.T).view(batch_size,batch_size)
            a_logits_per_phar_2_aug = (logit_scale * phar_features_2_all  @ smiles_features_aug_all.T).view(batch_size,batch_size)

            a_logits_per_smiles_1 = (logit_scale * smiles_features_all @ phar_features_1_all.T).view(batch_size,batch_size)
            a_logits_per_smiles_2 = (logit_scale * smiles_features_all @ phar_features_2_all.T).view(batch_size,batch_size)
            a_logits_per_smiles_1_aug = (logit_scale * smiles_features_aug_all @ phar_features_1_all.T).view(batch_size,batch_size)
            a_logits_per_smiles_2_aug = (logit_scale * smiles_features_aug_all @ phar_features_2_all.T).view(batch_size,batch_size)

            phar_features_1 = (v_c / (v_c.norm(dim=-1, keepdim=True)+1e-10))
            phar_features_2 = (v2_c / (v2_c.norm(dim=-1, keepdim=True)+1e-10))
            smiles_features = (zz / (zz.norm(dim=-1, keepdim=True)+1e-10))
            smiles_features_aug = (zz2 / (zz2.norm(dim=-1, keepdim=True)+1e-10))

            def masked_similarity(v_c, smiles_features, logit_scale=1.0):
                v_c = v_c.permute(1, 0, 2)
                smiles_features = smiles_features.permute(1, 0, 2)
               
                vc_exp = v_c.unsqueeze(1)             
                sf_exp = smiles_features.unsqueeze(0) 

                sim_matrix_ = torch.einsum("bqld,brmd->brlm", vc_exp, sf_exp)
                sim_matrix = sim_matrix_ * logit_scale
                sim_final = sim_matrix.max(dim=-1)[0].mean(dim=-1) 
                
                return sim_final

            logits_per_phar_1 = masked_similarity(phar_features_1, smiles_features,logit_scale=logit_scale2)
            logits_per_phar_2 = masked_similarity(phar_features_2, smiles_features,logit_scale=logit_scale2)
            logits_per_phar_1_aug = masked_similarity(phar_features_1, smiles_features_aug,logit_scale=logit_scale2)
            logits_per_phar_2_aug = masked_similarity(phar_features_2, smiles_features_aug,logit_scale=logit_scale2)

            logits_per_smiles_1 = masked_similarity(smiles_features, phar_features_1,logit_scale=logit_scale2)
            logits_per_smiles_2 = masked_similarity(smiles_features, phar_features_2,logit_scale=logit_scale2)
            logits_per_smiles_1_aug = masked_similarity(smiles_features_aug, phar_features_1,logit_scale=logit_scale2)
            logits_per_smiles_2_aug = masked_similarity(smiles_features_aug, phar_features_2,logit_scale=logit_scale2)
            
            _, N_frag, _ = fi.shape
            D = zz.shape[-1]
            smiles_features_f = zz.permute(1, 0, 2)
            frag_idx_clamped = fi.clamp(min=0)
            frag_mask = (fi != -1).float()
            frag_idx_expanded = frag_idx_clamped.unsqueeze(-1).expand(-1, -1, -1, D)
            token_selected = torch.gather(smiles_features_f.unsqueeze(1).expand(-1, N_frag, -1, -1), 2, frag_idx_expanded)

            token_selected = token_selected * frag_mask.unsqueeze(-1) 
            frag_embeddings = token_selected.sum(dim=2) / (frag_mask.sum(dim=2, keepdim=True) + 1e-8)  
            frag_embeddings_i = (frag_embeddings / (frag_embeddings.norm(dim=-1, keepdim=True)+1e-10))

            _, N_frag, _ = ft.shape
            D = zz2.shape[-1]
            smiles_features_f = zz2.permute(1, 0, 2)
            frag_idx_clamped = ft.clamp(min=0)
            frag_mask = (ft != -1).float() 
            frag_idx_expanded = frag_idx_clamped.unsqueeze(-1).expand(-1, -1, -1, D)
            token_selected = torch.gather(smiles_features_f.unsqueeze(1).expand(-1, N_frag, -1, -1), 2, frag_idx_expanded)
            token_selected = token_selected * frag_mask.unsqueeze(-1) 
            frag_embeddings = token_selected.sum(dim=2) / (frag_mask.sum(dim=2, keepdim=True) + 1e-8)
            frag_embeddings_t = (frag_embeddings / (frag_embeddings.norm(dim=-1, keepdim=True)+1e-10))

            f_logits_per_phar_1 = masked_similarity(phar_features_1, frag_embeddings_i.permute(1, 0, 2),logit_scale=logit_scale3)
            f_logits_per_phar_2 = masked_similarity(phar_features_2, frag_embeddings_i.permute(1, 0, 2),logit_scale=logit_scale3)
            f_logits_per_phar_1_aug = masked_similarity(phar_features_1, frag_embeddings_t.permute(1, 0, 2),logit_scale=logit_scale3)
            f_logits_per_phar_2_aug = masked_similarity(phar_features_2, frag_embeddings_t.permute(1, 0, 2),logit_scale=logit_scale3)

            f_logits_per_smiles_1 = masked_similarity(frag_embeddings_i.permute(1, 0, 2), phar_features_1,logit_scale=logit_scale3)
            f_logits_per_smiles_2 = masked_similarity(frag_embeddings_i.permute(1, 0, 2), phar_features_2,logit_scale=logit_scale3)
            f_logits_per_smiles_1_aug = masked_similarity(frag_embeddings_t.permute(1, 0, 2), phar_features_1,logit_scale=logit_scale3)
            f_logits_per_smiles_2_aug = masked_similarity(frag_embeddings_t.permute(1, 0, 2), phar_features_2,logit_scale=logit_scale3)

            if if_gene:
                zzz, encoder_mask = self.expand_then_fusing(z, vm2, vvs2) 
                gene_smiles = targets
                _, gene_length = gene_smiles.shape
                gene_mask = torch.triu(torch.ones(gene_length, gene_length, dtype=torch.bool),
                                        diagonal=1).to(gene_smiles.device)
                gene_embed = self.smiencoder.word_embed(gene_smiles)
                gene_embed = self.smiencoder.pos_encoding(gene_embed.permute(1, 0, 2).contiguous())

                output_gene = self.decoder(gene_embed, zzz,
                                    x_mask=gene_mask, mem_padding_mask=encoder_mask).permute(1, 0, 2).contiguous()
                gene_prediction_scores = self.smiencoder.word_pred(output_gene) 

                gene_shifted_prediction_scores = gene_prediction_scores[:, :-1, :].contiguous()
                gene_lm_loss = F.cross_entropy(gene_shifted_prediction_scores.view(-1, self.smiencoder.vocab_size), gene_smiles[:, 1:].contiguous().view(-1),
                                        ignore_index=self.smiencoder.pad_value)
                

            if return_dict:
                ret_dict = {}
                ret_dict['logits'] = logits_per_phar_1, logits_per_phar_2, logits_per_smiles_1, logits_per_smiles_2
                ret_dict['logits_aug'] = logits_per_phar_1_aug, logits_per_phar_2_aug, logits_per_smiles_1_aug, logits_per_smiles_2_aug
                ret_dict['logits_aug_a'] = a_logits_per_phar_1_aug, a_logits_per_phar_2_aug, a_logits_per_smiles_1_aug, a_logits_per_smiles_2_aug, a_logits_per_phar_1, a_logits_per_phar_2, a_logits_per_smiles_1, a_logits_per_smiles_2
                ret_dict['logits_aug_f'] = f_logits_per_phar_1_aug, f_logits_per_phar_2_aug, f_logits_per_smiles_1_aug, f_logits_per_smiles_2_aug, f_logits_per_phar_1, f_logits_per_phar_2, f_logits_per_smiles_1, f_logits_per_smiles_2
                ret_dict['features'] = smiles_features, phar_features_1, phar_features_2
                ret_dict['z_klloss'] = kl_loss_z
                ret_dict['lm_loss'] = lm_loss
                if if_gene:
                    ret_dict['gene_lm_loss'] = gene_lm_loss

                return ret_dict
       