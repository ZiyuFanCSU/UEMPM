import random
from collections import defaultdict
from typing import List, Set, Union, Dict
import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm
from collections import defaultdict

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


def process_smiles(index_smiles_pair):
    i, smiles = index_smiles_pair
    scaffold = generate_scaffold(smiles, include_chirality=True)
    return scaffold, i



def scaffold_to_smiles(mols: Union[List[str], List[Chem.Mol]],
                       use_indices: bool = False) -> Dict[str, Union[Set[str], Set[int]]]:
    """
    Computes scaffold for each smiles string and returns a mapping from scaffolds to sets of smiles.

    :param mols: A list of smiles strings or RDKit molecules.
    :param use_indices: Whether to map to the smiles' index in all_smiles rather than mapping
    to the smiles string itself. This is necessary if there are duplicate smiles.
    :return: A dictionary mapping each unique scaffold to all smiles (or smiles indices) which have that scaffold.
    """
    scaffolds = defaultdict(set)
    for i, mol in tqdm(enumerate(mols), total=len(mols)):
        if Chem.MolFromSmiles(mol) != None:
            scaffold = generate_scaffold(mol)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)

    return scaffolds


def scaffold_split(index, smiles_list,
                   frac_train=0.8, frac_valid=0.1, frac_test=0.1,
                   balanced: bool = False,
                   seed: int = 0):

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
    index = np.array(index)

    # Split
    train_size, val_size, test_size = frac_train * len(smiles_list), frac_valid * len(smiles_list), frac_test * len(
        smiles_list)
    train, val, test = [], [], []
    train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

    # Map from scaffold to index in the smiles
    scaffold_to_indices = scaffold_to_smiles(smiles_list, use_indices=True)

    if balanced:  # Put stuff that's bigger than half the val/test size into train, rest just order randomly
        index_sets = list(scaffold_to_indices.values())
        big_index_sets = []
        small_index_sets = []
        for index_set in index_sets:
            if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                big_index_sets.append(index_set)
            else:
                small_index_sets.append(index_set)
        random.seed(seed)
        random.shuffle(big_index_sets)
        random.shuffle(small_index_sets)
        index_sets = big_index_sets + small_index_sets
    else:  # Sort from largest to smallest scaffold sets
        index_sets = sorted(list(scaffold_to_indices.values()), key=lambda index_set: len(index_set), reverse=True)

    for index_set in index_sets:
        if len(train) + len(index_set) <= train_size:
            train += index_set
            train_scaffold_count += 1
        elif len(val) + len(index_set) <= val_size:
            val += index_set
            val_scaffold_count += 1
        else:
            test += index_set
            test_scaffold_count += 1
    print(
        f'Total scaffolds = {len(scaffold_to_indices)} | train scaffolds = {train_scaffold_count} | val scaffolds = {val_scaffold_count} | test scaffolds = {test_scaffold_count}')

    train_idx = index[train]
    val_idx = index[val]
    test_idx = index[test]

    return train_idx, val_idx, test_idx