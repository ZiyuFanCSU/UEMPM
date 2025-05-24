import re
from typing import List
import torch.nn.functional as F
import dgl
import numpy as np
import torch
from rdkit import Chem
# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import pickle
import random
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
# from logging import getLogger
# LOGGER = getLogger('main')

class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>') # SPECIAL_TOKENS 包含了特殊的 token，如 <sos>（开始符号）、<eos>（结束符号）、<pad>（填充符号）、<mask>（掩码符号）、<sep>（分隔符号）、<unk>（未知符号），以及额外保留的 <t_i> 格式的特殊 token。
    SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use
    # PATTEN 是一个正则表达式，用于匹配 SMILES 中的不同类型的子字符串，如原子、特殊符号等。
    PATTEN = re.compile(r'\[[^\]]+\]'  
                        # only some B|C|N|O|P|S|F|Cl|Br|I atoms can omit square brackets
                        r'|B[r]?|C[l]?|N|O|P|S|F|I'
                        r'|[bcnops]'
                        r'|@@|@'
                        r'|%\d{2}'
                        r'|.')
    # ATOM_PATTEN 是用于匹配原子的子字符串的正则表达式。
    ATOM_PATTEN = re.compile(r'\[[^\]]+\]'
                             r'|B[r]?|C[l]?|N|O|P|S|F|I'
                             r'|[bcnops]')

    @staticmethod
    def gen_vocabs(smiles_list):
        smiles_set = set(smiles_list)
        vocabs = set()

        for a in tqdm(smiles_set):
            vocabs.update(re.findall(Tokenizer.PATTEN, a)) 

        return vocabs # 词汇表，这个词汇表比较简单

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
                a = 3  # 3 for <mask> !!!!!!
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


def run_test_tokenizer():
    smiles = ['CCNC(=O)NInc1%225cpppcc2nc@@nc(N@c3ccc(O[C@@H+5]c4cccc(F-)c4)c(Cl)c3)c2c1']
    tokenizer = Tokenizer(Tokenizer.gen_vocabs(smiles))
    print(tokenizer.parse(smiles[0]))
    print(tokenizer.get_text([tokenizer.parse(smiles[0])]))


def _corrupt(token_seq: List[int], mask_token, corrupt_percent=0.1, poisson_lambda=2): # mask序列
    # infilling, not perfect
    token_seq = token_seq.copy()
    l = len(token_seq)
    n = int(l * corrupt_percent)

    c = 0
    idx = sorted(np.random.choice(list(range(1, l - 1)), n), reverse=True)  # skip <sos>
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
    """
    计算两个 SMILES 之间的 Tanimoto 相似度。
    
    参数:
    - smiles1: 第一个 SMILES 字符串
    - smiles2: 第二个 SMILES 字符串
    
    返回:
    - Tanimoto 相似度
    """
    # 将 SMILES 转换为 RDKit 分子对象
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    # 生成 MACCS 指纹
    fp1 = MACCSkeys.GenMACCSKeys(mol1)
    fp2 = MACCSkeys.GenMACCSKeys(mol2)
    
    # 计算 Tanimoto 系数
    tanimoto = DataStructs.FingerprintSimilarity(fp1, fp2)
    
    return tanimoto

def calculate_tanimoto_similarity_matrix(smiles_list):
    """
    计算 SMILES 列表中的两两 Tanimoto 相似度，返回 0/1 矩阵。
    
    参数:
    - smiles_list: SMILES 字符串列表
    
    返回:
    - 0/1 相似度矩阵 (numpy array)
    """
    num_smiles = len(smiles_list)
    similarity_matrix = np.zeros((num_smiles, num_smiles), dtype=int)
    # 逐对计算 SMILES 之间的 Tanimoto 相似度
    for i in range(num_smiles):
        for j in range(i + 1, num_smiles):
            sim = tanimoto_similarity(smiles_list[i], smiles_list[j])
            
            # 如果 Tanimoto 相似度大于 0.9，则矩阵值为 1，否则为 0
            similarity_matrix[i, j] = 1 if sim > 0.85 else 0
            similarity_matrix[j, i] = similarity_matrix[i, j]  # 矩阵是对称的
    np.fill_diagonal(similarity_matrix, 1)
    return torch.tensor(similarity_matrix)

class SemiSmilesDataset(Dataset):

    def __init__(self, data,labels, tokenizer: Tokenizer):
        """
        :param smiles_list: list of valid smiles
        :param tokenizer:
        :param use_random_input_smiles:
        :param use_random_target_smiles:
        :param rsmiles:
        :param corrupt: boolean, whether to use infilling scheme to corrupt input smiles
        """
        super().__init__()
           
        self.data = data
        self.label = labels
        
        self.tokenizer = tokenizer
        self.mask_token = tokenizer.SPECIAL_TOKENS.index('<mask>') # 3

        self.vocab_size = len(tokenizer)
        self.len = len(self.data)
        
        self.device = "cpu"


    def __len__(self):
        return self.len

    def convert_string(self,s):
        # 替换中文逗号为英文逗号，然后按照英文逗号分割字符串
        elements = s.split(',')
        
        # 将第一个元素保持为字符串，其他元素转换为 float
        result = [elements[0]] + [float(e) for e in elements[1:]]
        
        return result

    def __getitem__(self, item):
        smiles = self.convert_string(self.data[item])[0]
        label = self.label[item]
        mol = Chem.MolFromSmiles(smiles)
        
        # clear isotope
        for atom in mol.GetAtoms(): # 清除了分子中的同位素信息
            atom.SetIsotope(0)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol)) # 规范分子的表示方式
        # 转换为规范化的 SMILES 字符串，不包含立体异构信息
        input_smiles = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True, doRandom=False)
        input_seq,atom_idx_i = self.tokenizer.parse(input_smiles, return_atom_idx=True)

        return input_seq,label,input_smiles,smiles

    @staticmethod # 用于将一个 batch 的样本进行处理和批次化
    def collate_fn(batch):
        pad_token = Tokenizer.SPECIAL_TOKENS.index('<pad>') # 2

        corrupted_inputs,label,input_smiles,yuan = list(zip(*batch))

        corrupted_inputs = \
            pad_sequences(corrupted_inputs, batch_first=True, padding_value=pad_token)
        input_mask = (corrupted_inputs==pad_token).bool()

        return corrupted_inputs, input_mask, torch.stack([torch.tensor(l) for l in label]), list(yuan)


def pad_sequences(sequences, batch_first=True, padding_value=0, lif = 128):
    # 计算每个序列的长度
    lengths = [len(seq) for seq in sequences]
    if lif == -1:
        # 找到最大长度
        max_length = max(lengths)
    else:
        max_length = lif
    
    # 初始化一个张量，大小为 (batch_size, max_length)
    if batch_first:
        padded_sequences = torch.full((len(sequences), max_length), padding_value)
    else:
        padded_sequences = torch.full((max_length, len(sequences)), padding_value)
    
    # 填充每个序列
    # 填充或截断序列
    for i, seq in enumerate(sequences):
        truncated_seq = seq[:max_length]  # 截断超长序列
        truncated_seq = torch.tensor(truncated_seq, dtype=padded_sequences.dtype)

        if batch_first:
            padded_sequences[i, :len(truncated_seq)] = truncated_seq
        else:
            padded_sequences[:len(truncated_seq), i] = truncated_seq
    
    return padded_sequences
