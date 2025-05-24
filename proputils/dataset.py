import re
from typing import List
import torch.nn.functional as F
import numpy as np
import torch
from rdkit import Chem
from torch.nn.utils.rnn import pad_sequence as true_pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pickle
import random
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>') 
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  
    PATTEN = re.compile(r'\[[^\]]+\]'  
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a)) 

        return vocabs 

    def __init__(self, vocabs):
        special_tokens = list(Tokenizer.SPECIAL_TOKENS)
        vocabs = special_tokens + sorted(set(vocabs) - set(special_tokens), key=lambda x: (len(x), x))
        self.vocabs = vocabs
        self.i2s = {i: s for i, s in enumerate(vocabs)}
        self.s2i = {s: i for i, s in self.i2s.items()}

    def __len__(self):
        return len(self.vocabs)

    def parse(self, smiles, return_atom_idx=False):
        l = []
        if return_atom_idx:
            atom_idx=[]
        for i, s in enumerate(('<sos>', *re.findall(Tokenizer.PATTEN, smiles), '<eos>')):
            if s not in self.s2i:
                a = 3 
            else:
                a = self.s2i[s]
            l.append(a)
            
            if return_atom_idx and re.fullmatch(Tokenizer.ATOM_PATTEN, s) is not None:
                atom_idx.append(i)
        if return_atom_idx:
            return l, atom_idx
        return l

    def get_text(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.tolist()

        smiles = []
        for p in predictions:
            s = []
            for i in p:
                c = self.i2s[i]
                if c == '<eos>':
                    break
                s.append(c)
            smiles.append(''.join(s))
        return smiles

def pad_sequences(sequences, batch_first=True, padding_value=0, lif = 128):
    lengths = [len(seq) for seq in sequences]
    if lif == -1:
        max_length = max(lengths)
    else:
        max_length = lif
    
    if batch_first:
        padded_sequences = torch.full((len(sequences), max_length), padding_value)
    else:
        padded_sequences = torch.full((max_length, len(sequences)), padding_value)
    
    for i, seq in enumerate(sequences):
        if batch_first:
            padded_sequences[i, :lengths[i]] = seq
        else:
            padded_sequences[:lengths[i], i] = seq
    return padded_sequences

def _corrupt(token_seq: List[int], mask_token, corrupt_percent=0.1, poisson_lambda=2): # mask序列
    token_seq = token_seq.copy()
    l = len(token_seq)
    n = int(l * corrupt_percent)

    c = 0
    idx = sorted(np.random.choice(list(range(1, l - 1)), n), reverse=True)
    for i in idx:
        li = np.random.poisson(poisson_lambda)
        while li < 1:
            li = np.random.poisson(poisson_lambda)
        token_seq[i] = mask_token
        li -= 1
        p = i + 1
        while p < l and li > 0:
            del token_seq[p]
            l -= 1
            li -= 1
            c += 1
        if c >= n:
            break
    return token_seq


def get_random_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)

    # clear isotope
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))

    rsmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)

    return rsmiles

def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    
    tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
    
    return tanimoto

def calculate_tanimoto_similarity_matrix(smiles_list):

    num_smiles = len(smiles_list)
    similarity_matrix = np.zeros((num_smiles, num_smiles), dtype=int)
    for i in range(num_smiles):
        for j in range(i + 1, num_smiles):
            sim = tanimoto_similarity(smiles_list[i], smiles_list[j])
            similarity_matrix[i, j] = 1 if sim > 0.9 else 0
            similarity_matrix[j, i] = similarity_matrix[i, j]  
    np.fill_diagonal(similarity_matrix, 1)
    return torch.tensor(similarity_matrix)


class SemiSmilesDataset(Dataset):

    def __init__(self, datapath, tokenizer: Tokenizer,
                 use_random_input_smiles=False, use_random_target_smiles=False, rsmiles=None, corrupt=True,shuffle = True,nogeneds=False,partone = False,partvalue = None,normal_path = None):
        """
        :param smiles_list: list of valid smiles
        :param tokenizer:
        :param use_random_input_smiles:
        :param use_random_target_smiles:
        :param rsmiles:
        :param corrupt: boolean, whether to use infilling scheme to corrupt input smiles
        """
        super().__init__()
        with open(normal_path, 'rb') as w:
            norm = pickle.load(w)
        self.property_mean = list(norm['mean'])
        self.property_std = list(norm['variance'])
        self.part_one = partone
        with open(datapath, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
        self.data = [l.strip() for l in lines]
        
        self.part_value = partvalue
    
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.SPECIAL_TOKENS.index('<mask>') # 3

        self.vocab_size = len(tokenizer)
        self.len = len(self.data)
        
        self.use_random_input_smiles = use_random_input_smiles
        self.use_random_target_smiles = use_random_target_smiles
        self.nogeneds = nogeneds
        self.rsmiles = rsmiles
        self.corrupt = corrupt
        self.device = "cpu"
        if shuffle:
            random.shuffle(self.data)

        if rsmiles is None and (use_random_input_smiles or use_random_target_smiles):
            print('WARNING: The result of rdkit.Chem.MolToSmiles(..., doRandom=True) is NOT reproducible '
                  'because this function does not provide a way to control its random seed.')


    def __len__(self):
        return self.len

    def convert_string(self,s):
        elements = s.split(',')
        result = [elements[0]] + [float(e) for e in elements[1:]]
        return result

    def __getitem__(self, item):
        smiles = self.convert_string(self.data[item])[0]
        mol = Chem.MolFromSmiles(smiles)
        
        # clear isotope
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) 
        csmiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, doRandom=False)
        rsmiles1 = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)
        rsmiles2 = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=False, doRandom=True)
        input_smiles = rsmiles1 if self.use_random_input_smiles else csmiles 
        target_smiles = rsmiles2 if self.use_random_target_smiles else csmiles 

        if self.nogeneds:
            input_smiles = rsmiles1
            target_smiles = rsmiles1
        
        input_seq,atom_idx_i = self.tokenizer.parse(input_smiles, return_atom_idx=True)
        input_seq_frag = self.tokenizer.map_brics_atom_indices_to_token_indices(input_smiles,atom_idx_i)
        target_seq, atom_idx_t = self.tokenizer.parse(target_smiles, return_atom_idx=True)
        target_seq_frag = self.tokenizer.map_brics_atom_indices_to_token_indices(target_smiles,atom_idx_t)
        
        if self.corrupt:
            corrupted_input = _corrupt(input_seq, self.mask_token) # mask
        else:
            corrupted_input = input_seq
        
        corrupted_input = torch.LongTensor(corrupted_input)
        target_seq = torch.LongTensor(target_seq)
        part1 = torch.tensor(self.convert_string(self.data[item])[1:], dtype=torch.float)

        properties = (part1 - torch.tensor(self.property_mean, dtype=torch.float)) / torch.tensor(self.property_std, dtype=torch.float)

        return corrupted_input, target_seq, properties, smiles,self.data[item],input_smiles,target_smiles,input_seq_frag,target_seq_frag

    @staticmethod 
    def collate_fn(batch):
        pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>') # 2

        corrupted_inputs, target_seqs, props, smi,true_p,input_smiles,target_smiles,input_seq_frag,target_seq_frag, *other_descriptors = list(zip(*batch))

        corrupted_inputs = \
            pad_sequences(corrupted_inputs, batch_first=True, padding_value=pad_token)
        input_mask = (corrupted_inputs==pad_token).bool()

        target_seqs = pad_sequences(target_seqs, batch_first=True, padding_value=pad_token)
        target_mask = (target_seqs==pad_token).bool()

        max_frag_len = max(len(frag) for sample in input_seq_frag for frag in sample)
        batched_samples = []
        for sample in input_seq_frag:
            frag_tensors = [
                F.pad(torch.tensor(frag, dtype=torch.long), (0, max_frag_len - len(frag)), value=-1)
                for frag in sample
            ]
            sample_tensor = torch.stack(frag_tensors, dim=0)  
            batched_samples.append(sample_tensor)

        input_seq_frag = true_pad_sequence(batched_samples, batch_first=True, padding_value=-1)
        max_frag_len = max(len(frag) for sample in target_seq_frag for frag in sample)
        batched_samples = []
        for sample in target_seq_frag:
            frag_tensors = [
                F.pad(torch.tensor(frag, dtype=torch.long), (0, max_frag_len - len(frag)), value=-1)
                for frag in sample
            ]
            sample_tensor = torch.stack(frag_tensors, dim=0) 
            batched_samples.append(sample_tensor)

        target_seq_frag = true_pad_sequence(batched_samples, batch_first=True, padding_value=-1)

        return corrupted_inputs, input_mask, target_seqs,target_mask,torch.stack(props),calculate_tanimoto_similarity_matrix(list(smi)),list(smi),true_p,list(input_smiles),list(target_smiles),input_seq_frag,target_seq_frag

