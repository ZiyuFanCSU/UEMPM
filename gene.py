import argparse
import pickle
import rdkit
import torch
from rdkit import RDLogger
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import DataLoader
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
from stutils.tokenizer import Tokenizer
from proputils.gene_model import geneModel
from proputils.utils import *
from proputils.dataset import SemiSmilesDataset
from stutils.STmodel import *
import pickle
import argparse
from rdkit.DataStructs import FingerprintSimilarity
import numpy as np
from utils.evaluate import average_agg_tanimoto,compute_fragments,compute_scaffolds,fingerprints,cos_similarity
import numpy as np
import pickle
from utils.loss import *
from utils.evaluate import calculate_molecular_properties
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.Scaffolds import MurckoScaffold   
from rdkit.Chem import AllChem
from collections import defaultdict 
from multiprocessing import Pool

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gene_N', type=int, default=20)
    parser.add_argument('--file_num', type=int, default=1)
    parser.add_argument('--normal_path', type=str, default="./datasets/normalize.pkl")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--k', type=int, default=2)

    args = parser.parse_args()
    print(args)
    return args

def load_model(address,device,st_tokenizer):
    args = parse_args()

    ST_MODEL_DEFAULT_SETTINGS = {
    "max_len": 128,  
    "pp_v_dim": 8,  
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
    "bs":args.batch_size
    }
    model_params = dict(ST_MODEL_DEFAULT_SETTINGS)

    phargnn_model = geneModel(model_params,st_tokenizer)
    state_dict = torch.load(address, map_location=device)

    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '') 
        new_state_dict[new_key] = state_dict[key]

    phargnn_model.load_state_dict(new_state_dict, strict=False)
    torch.save(phargnn_model.state_dict(), '/ifs/home/fanziyu/project/clipUEMPM/pretrained_models/gene.pth')
    phargnn_model.eval()
    return phargnn_model


def format_smiles(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        return None,None

    smiles = rdkit.Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True, doRandom=False)

    return smiles,mol


def _generate(device,incremental_state, model, props_embeds, text_embed, stochastic=False, prop_att_mask=None, kkkk=None):
        batch_size = 1

        token = torch.full((batch_size, model.smiencoder.max_len), model.smiencoder.pad_value, dtype=torch.long, device=device)
        token[:, 0] = model.smiencoder.sos_value

        text_pos = model.smiencoder.pos_encoding.pe

        if not stochastic:
            scores = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for t in range(1, model.smiencoder.max_len):
            one = model.decoder.forward_one(text_embed, props_embeds,incremental_state, mem_padding_mask=prop_att_mask)
            one = one.squeeze(0)

            l = model.smiencoder.word_pred(one) 
            if not stochastic:
                scores.append(l)
            if stochastic:
                k = torch.multinomial(torch.softmax(l, 1), 1).squeeze(1)
            else:
                values,indices = torch.topk(torch.softmax(l, dim=-1), k=kkkk, dim=-1)
                sampled_indices = torch.multinomial(values, num_samples=1)
                kk = indices.gather(1, sampled_indices)[0]
            token[:, t] = kk

            finished |= kk == model.smiencoder.eos_value
            if finished.all():
                break

            text_embed = model.smiencoder.word_embed(kk)
            text_embed = text_embed + text_pos[t]  #
            text_embed = text_embed.unsqueeze(0)

        predict = token[:, 1:]

        if not stochastic:
            return st_tokenizer.get_text(predict), torch.stack(scores, dim=1)
        return predict

def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain assert from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold    
def process_smiles(smiles):
    scaffold = generate_scaffold(smiles, include_chirality=True)
    return smiles, scaffold    
def build_scaffolds(smiles_list, num_workers=15):
    scaffolds = defaultdict(list)
    with Pool(num_workers) as pool:
        results = pool.map(process_smiles, smiles_list)
    for ind, (_, scaffold) in enumerate(results):
        if scaffold:  
            scaffolds[scaffold].append(ind)
    return scaffolds    
def get_scaffoldnum(all_smiles):
    scaffolds = defaultdict(list)

    scaffolds = build_scaffolds(all_smiles, num_workers=8)

    re = len(set(list(scaffolds.keys())))
    return re
def get_ecfp6_fingerprints(mols,r=3, include_none=False):
    """
    Get ECFP6 fingerprints for a list of molecules. Optionally,
    handle `None` values by returning a `None` value in that
    position.
    """
    fps = []
    for mol in mols:
        if mol is None and include_none:
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=1024)
            fps.append(fp)
    return(fps)
def internal_diversity(fps, sample_size=10000, summarise=True):
    """
    Calculate the internal diversity, defined as the mean intra-set Tanimoto
    coefficient, between a set of fingerprints. For large sets, calculating the
    entire matrix is prohibitive, so a random set of molecules are sampled.
    """
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = random.randint(0, len(fps) - 1)
        idx2 = random.randint(0, len(fps) - 1)
        fp1 = fps[idx1]
        fp2 = fps[idx2]
        tcs.append(1-FingerprintSimilarity(fp1, fp2))
        counter += 1
    if summarise:
        return np.mean(tcs)
    else:
        return tcs
def external_diversity(fps1, fps2, sample_size=10000, summarise=True):
    """
    Calculate the external diversity, defined as the mean inter-set Tanimoto
    coefficient, between two sets of fingerprints. For large sets, calculating
    the entire matrix is prohibitive, so a random set of molecules are sampled.
    """
    #
    tcs = []
    counter = 0
    while counter < sample_size:
        idx1 = random.randint(0, len(fps1) - 1)
        idx2 = random.randint(0, len(fps2) - 1)
        fp1 = fps1[idx1]
        fp2 = fps2[idx2]
        tcs.append(1-FingerprintSimilarity(fp1, fp2))
        counter += 1
    if summarise:
        return np.mean(tcs)
    else:
        return tcs


def eval(sms,pp,tt):
    v_pp = []
    valid_mol = []
    for i in pp:
        v_s,v_mol = format_smiles(i)
        v_pp.append(v_s)
        if v_mol != None:
            valid_mol.append(v_mol)

    valid_smiles = [i for i in v_pp if i is not None]
    s_valid_smiles = set(valid_smiles)
    uniqueness = len(s_valid_smiles) / len(valid_smiles)
    novelty = len(s_valid_smiles - set(tt)) / len(s_valid_smiles)
    available = len(s_valid_smiles - set(tt)) / len(v_pp)
    validity = len(valid_smiles) / len(v_pp)
    print(f'Validity: {validity:.4f} Uniqueness: {uniqueness:.4f} Novelty: {novelty:.4f} Available: {available:.4f} ')
    
    # RMSD
    df_train = pd.read_csv('./datasets/s_my_train_53prop.csv')
    df_train = df_train.drop(columns=['smiles'])
    train_mean = df_train.mean()
    train_std = df_train.std()
    canzhao_values = calculate_molecular_properties(sms[0])[1] 

    gene_prop = []
    for i in valid_smiles:
        gene_prop.append(calculate_molecular_properties(i)[1])
    df = pd.DataFrame(gene_prop, columns=df_train.columns)
    df_normalized = (df - train_mean) / train_std
    canzhao_df_normalized = (pd.DataFrame([canzhao_values] * len(df), columns=df.columns) - train_mean) / train_std
    rmsd = np.sqrt(((df_normalized - canzhao_df_normalized) ** 2).mean())
    nrmse_sum = rmsd.sum()
    nrmse_mean = rmsd.mean()
    scanum = get_scaffoldnum(valid_smiles)
    print(f'nrmse_sum:{nrmse_sum:.4f} nrmse_mean:{nrmse_mean:.4f} scaffolds:{scanum}')
    
 

@torch.no_grad()
def evaluate(args,vocab,model, data_loader, device, stochastic=False, k=2):
    # test
    print(f"PV-to-SMILES generation in {'stochastic' if stochastic else 'deterministic'} manner with k={k}...")
    model.eval()
    
    for _, batch_data in enumerate(data_loader):

        gene = []
        inputs = batch_data[0].to(device)
        batch_size = inputs.shape[0]
        props = batch_data[4].to(inputs.device, non_blocking=True)
        print(batch_data[7])
        sms = batch_data[6]

        v,vm,vvs = model.process_p_n(props,value = 0)
        v = v.unsqueeze(0)

        for _ in range(args.gene_N):
            z = torch.randn(1,batch_size,model.hidden_dim).to(device)
            zzz, encoder_mask = model.expand_then_fusing(z, vm, vvs)

            token = torch.full((batch_size, model.smiencoder.max_len), model.smiencoder.pad_value, dtype=torch.long, device=device)
            token[:, 0] = model.smiencoder.sos_value
            text_pos = model.smiencoder.pos_encoding.pe
            text_embed = model.smiencoder.word_embed(token[:, 0])
            text_embed = text_embed + text_pos[0]
            text_embed = text_embed.unsqueeze(0)
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[torch.Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
            )
            pp,_ = _generate(device,incremental_state, model, zzz, text_embed, stochastic=stochastic, prop_att_mask=encoder_mask, kkkk=k)
            gene += pp

        eval(sms,gene,vocab)
        print("=" * 50)



if __name__ == '__main__':
    args = parse_args()
    with open('./datasets/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    st_tokenizer = Tokenizer(Tokenizer.gen_vocabs(vocab)) # 建立的词汇表
    device = "cuda:3"  #
    model = load_model('./pretrained_models/gene.pth',device,st_tokenizer)
    
    model.eval()
    model.to(device)
    gene_dataset = SemiSmilesDataset("./datasets/gene.csv", st_tokenizer,
                                    use_random_input_smiles = False, use_random_target_smiles = False, corrupt=False,shuffle = False
                                    ,normal_path=args.normal_path)
    
    gene_sampler = RandomSampler(gene_dataset)
    gene_loader = DataLoader(gene_dataset,
                              batch_size=args.batch_size,
                              sampler=gene_sampler,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=gene_dataset.collate_fn,
                              persistent_workers=True)
    
    evaluate(args,vocab,model, gene_loader, device, stochastic=False, k=args.k)
    
    



