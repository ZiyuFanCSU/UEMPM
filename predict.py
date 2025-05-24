import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader
from enum import Enum
from tqdm import tqdm
import time
from rdkit import RDLogger
from collections import Counter
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
from stutils.tokenizer import Tokenizer
from proputils.pt_model import PropModel
from proputils.utils import *
from proputils.dataset_pred import SemiSmilesDataset
from stutils.STmodel import *
from utils.splitter import scaffold_split
import pickle
import argparse
from proputils.model import MModel
import numpy as np
import pickle
from utils.loss import *
import pandas as pd
from utils.cal import train_one_epoch_multitask,evaluate_on_multitask

def load_smiles(txt_file, task_type="classification"):
    '''
    :param txt_file: should be {dataset}_processed_ac.csv
    :return:
    '''
    df = pd.read_csv(txt_file)
    smiles = df["smiles"].values.flatten().tolist()
    index = df["label"].values
    labels = np.array(df.label.apply(lambda x: str(x).split(' ')).tolist())
    labels = labels.astype(int) if task_type == "classification" else labels.astype(float)
    return smiles,labels


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', type=str, default='classification', help='classification/regression')
    parser.add_argument('--dataset', type=str, default='toxcast', help='esol/lipophilicity/bbbp/bace/tox21/clintox/toxcast')
    
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--iflock', type=bool, default=False)
    parser.add_argument('--save_finetune_ckpt', type=bool, default=0)
    parser.add_argument('--weighted_CE', action='store_true', default=False, help='whether to use global imbalanced weight')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', default=5e-5, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight decay pow (default: -5)')
    parser.add_argument('--momentum', default=0.9, type=float, help='moment um (default: 0.9)')

    parser.add_argument('--epochs', type=int, default=3000, help='number of total epochs to run (default: 100)')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--hidden_dim', type=int, default=384)

    args = parser.parse_args()
    print(args)
    return args

def is_left_better_right(left_num, right_num, standard):
    '''
    :param left_num:
    :param right_num:
    :param standard: if max, left_num > right_num is true, if min, left_num < right_num is true.
    :return:
    '''
    assert standard in ["max", "min"]
    if standard == "max":
        return left_num > right_num
    elif standard == "min":
        return left_num < right_num

def train(st_tokenizer,device):
    args = parse_args()
    if args.task_type == "classification":
        eval_metric = "rocauc"
        valid_select = "max"
        min_value = -np.inf
    elif args.task_type == "regression":
        if args.dataset == "qm7" or args.dataset == "qm8" or args.dataset == "qm9":
            eval_metric = "mae"
        else:
            eval_metric = "rmse"
        valid_select = "min"
        min_value = np.inf

    smiles, labels = load_smiles('./datasets/MoleculeNet/'+str(args.dataset)+'.csv', task_type=args.task_type)
    num_tasks = labels.shape[1]

    train_idx, val_idx, test_idx = scaffold_split(list(range(0, len(smiles))), smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=args.seed, balanced=True)

    smiles_train, smiles_val, smiles_test, labels_train, labels_val, labels_test = np.array(smiles)[train_idx], np.array(smiles)[val_idx], np.array(smiles)[
        test_idx], labels[train_idx], labels[val_idx], labels[test_idx]
    
    t_dataset = SemiSmilesDataset(smiles_train,labels_train, st_tokenizer)
    t_sampler = RandomSampler(t_dataset)
    t_loader = DataLoader(t_dataset, 
                              batch_size=args.batch_size,
                              sampler=t_sampler,
                              drop_last=False,
                              shuffle= False,
                              num_workers=args.num_workers, 
                              pin_memory=False, 
                              collate_fn=t_dataset.collate_fn,
                              persistent_workers=True) 
    v_dataset = SemiSmilesDataset(smiles_val,labels_val, st_tokenizer)
    v_sampler = RandomSampler(v_dataset)
    v_loader = DataLoader(v_dataset,
                              batch_size=args.batch_size,
                              sampler=v_sampler,
                              drop_last=False,
                              shuffle= False,
                              num_workers=args.num_workers, 
                              pin_memory=False,
                              collate_fn=v_dataset.collate_fn,
                              persistent_workers=True)
    s_dataset = SemiSmilesDataset(smiles_test,labels_test, st_tokenizer)
    s_sampler = RandomSampler(s_dataset)
    s_loader = DataLoader(s_dataset, 
                              batch_size=args.batch_size,
                              sampler=s_sampler,
                              drop_last=False,
                              shuffle= False,
                              num_workers=args.num_workers, 
                              pin_memory=False, 
                              collate_fn=s_dataset.collate_fn,
                              persistent_workers=True) 

    model = MModel(st_tokenizer,args.hidden_dim,num_tasks)
    
    if args.iflock:
        for param in model.model_pre.parameters():
            param.requires_grad = False
    
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=args.min_lr, last_epoch=-1)

    weights = None
    if args.task_type == "classification":
        if args.weighted_CE:
            labels_train_list = labels_train[labels_train != -1].flatten().tolist()
            count_labels_train = Counter(labels_train_list)
            imbalance_weight = {key: 1 - count_labels_train[key] / len(labels_train_list) for key in count_labels_train.keys()}
            weights = np.array(sorted(imbalance_weight.items(), key=lambda x: x[0]), dtype="float")[:, 1]
        criterion = nn.BCEWithLogitsLoss(reduction="none")
    elif args.task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise Exception("param {} is not supported.".format(args.task_type))

    ##################################### train #####################################
    results = {'highest_valid': min_value,
               'final_train': min_value,
               'final_test': min_value,
               'highest_train': min_value,
               'highest_valid_desc': None,
               "final_train_desc": None,
               "final_test_desc": None}

    early_stop = 0
    patience = 10

    for epoch in range(args.start_epoch, args.epochs):
        # train
        train_one_epoch_multitask(model=model, optimizer=optimizer, data_loader=t_loader, criterion=criterion,
                                  weights=weights, device=device, epoch=epoch, task_type=args.task_type)
        
        scheduler.step()
        # evaluate
        train_loss, train_results, _ = evaluate_on_multitask(model=model, data_loader=t_loader,
                                                                           criterion=criterion, device=device,
                                                                           epoch=epoch, task_type=args.task_type,
                                                                           return_data_dict=True)
        _, val_results, _ = evaluate_on_multitask(model=model, data_loader=v_loader,
                                                                     criterion=criterion, device=device,
                                                                     epoch=epoch, task_type=args.task_type,
                                                                     return_data_dict=True)
        _, test_results, _ = evaluate_on_multitask(model=model, data_loader=s_loader,
                                                                        criterion=criterion, device=device, epoch=epoch,
                                                                        task_type=args.task_type, return_data_dict=True)

        train_result = train_results[eval_metric.upper()]
        valid_result = val_results[eval_metric.upper()]
        test_result = test_results[eval_metric.upper()]

        print({"epoch": epoch, "patience": early_stop, "Loss": train_loss, 'Train': train_result,
               'Validation': valid_result, 'Test': test_result})

        if is_left_better_right(train_result, results['highest_train'], standard=valid_select):
            results['highest_train'] = train_result

        if is_left_better_right(valid_result, results['highest_valid'], standard=valid_select):
            results['highest_valid'] = valid_result
            results['final_train'] = train_result
            results['final_test'] = test_result

            results['highest_valid_desc'] = val_results
            results['final_train_desc'] = train_results
            results['final_test_desc'] = test_results

            early_stop = 0
        else:
            early_stop += 1
            if early_stop > patience:
                break

    print(f"{args.dataset}_{args.batch_size}_{args.lr:.6f}_{args.seed}")
    print("final results: highest_valid: {:.3f}, final_train: {:.3f}, final_test: {:.3f}"
          .format(results["highest_valid"], results["final_train"], results["final_test"]))

            
def main():
    with open('./datasets/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    st_tokenizer = Tokenizer(Tokenizer.gen_vocabs(vocab)) 
    train(st_tokenizer,device="cuda:2")

if __name__ == '__main__':

    main()
