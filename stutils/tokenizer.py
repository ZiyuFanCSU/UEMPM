import re
from typing import List
import numpy as np
import torch
from rdkit import Chem
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit.Chem.BRICS import BreakBRICSBonds
from rdkit.Chem import rdmolops


# from logging import getLogger
# LOGGER = getLogger('main')

class Tokenizer:
    NUM_RESERVED_TOKENS = 32
    SPECIAL_TOKENS = ('<sos>', '<eos>', '<pad>', '<mask>', '<sep>', '<unk>') # SPECIAL_TOKENS 包含了特殊的 token，如 <sos>（开始符号）、<eos>（结束符号）、<pad>（填充符号）、<mask>（掩码符号）、<sep>（分隔符号）、<unk>（未知符号），以及额外保留的 <t_i> 格式的特殊 token。
    # SPECIAL_TOKENS += tuple([f'<t_{i}>' for i in range(len(SPECIAL_TOKENS), 32)])  # saved for future use
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
    def smiles_to_brics_fragment_token_indices(self,smiles):
        """
        将 SMILES 按 BRICS 分片，并返回每个片段在原始 SMILES 中的原子（token）索引列表。
        Dummy atoms（[n*]）会自动排除。
        
        Returns:
            List[List[int]]: 每个子列表是一个 BRICS 片段对应的原始 token index。
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        # 1. 执行 BRICS 断键（保留原子索引，但插入 dummy atom）
        frag_mol = BreakBRICSBonds(mol)

        # 2. 找出所有 dummy atom 的原子索引（原子序号为 0）
        dummy_indices = set(
            atom.GetIdx() for atom in frag_mol.GetAtoms() if atom.GetAtomicNum() == 0
        )

        # 3. 获取分片（每片是一组原子 index，包含 dummy）
        frag_atom_ids = rdmolops.GetMolFrags(
            frag_mol, asMols=False, sanitizeFrags=False
        )

        # 4. 过滤掉 dummy atom，得到原始原子 token 索引
        clean_fragments = [
            [idx for idx in frag if idx not in dummy_indices]
            for frag in frag_atom_ids
        ]

        return clean_fragments
    
    def map_brics_atom_indices_to_token_indices(self,smilesss, atom_idx_seq):
        """
        将 BRICS 原子索引片段映射为 tokenizer parse 后的 token 索引。

        Args:
            brics_atom_fragments: List[List[int]]，每个元素是 BRICS 中一个片段的原子 index（mol 中）
            atom_idx_seq: List[int]，parse(smiles, return_atom_idx=True) 返回的 atom_idx，表示原子在 input_seq 中的位置

        Returns:
            List[List[int]]：每个片段对应的 token 索引
        """
        mapped_token_fragments = []
        brics_atom_fragments=self.smiles_to_brics_fragment_token_indices(smilesss)

        for frag in brics_atom_fragments:
            token_ids = []
            for atom_id in frag:
                if atom_id < len(atom_idx_seq):  # 保证原子索引在 tokenizer 范围内
                    token_ids.append(atom_idx_seq[atom_id])
                else:
                    print(f"[警告] 原子索引 {atom_id} 超出 tokenizer 处理范围！")
            mapped_token_fragments.append(token_ids)

        return mapped_token_fragments


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
                c = self.i2s[int(i)]
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

if __name__ == '__main__':
    run_test_tokenizer()
