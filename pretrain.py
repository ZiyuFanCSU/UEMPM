import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import DataLoader
import time
from rdkit import RDLogger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
import warnings
warnings.filterwarnings("ignore")
from stutils.tokenizer import Tokenizer
from proputils.pt_model import PropModel
from proputils.utils import *
from proputils.dataset import SemiSmilesDataset
from stutils.STmodel import *
from utils.splitter import *
import pickle
import argparse
import numpy as np
import numpy as np
import pickle
from utils.loss import *
from utils.misc import accuracy

class CFG:
    fp16 = False 
    print_freq = 200  
    num_workers = 1
    weight_decay = 1e-6
    min_lr = 5e-6  
    T_max = 4  

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='scaffold', help='random/scaffold/random_scaffold/scaffold_balanced')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--diff_datapath', type=str, default="./datasets/challenging.csv")
    parser.add_argument('--exam_datapath', type=str, default="./datasets/example.csv")
    parser.add_argument('--va_datapath', type=str, default="./datasets/valid.csv")
    parser.add_argument('--te_datapath', type=str, default="./datasets/test.csv")
    parser.add_argument('--normal_path', type=str, default="./datasets/normalize.pkl")
    parser.add_argument('--ddevice', type=str, default="cuda:0")
    parser.add_argument('--start_lr', type=float, default=5e-5)
    parser.add_argument('--kl_beta', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--model_type', choices=['rs_mapping', 'cs_mapping', 'non_vae', 'remove_pp_dis'], default='rs_mapping')
    parser.add_argument('--topk',type=int,  default=5)
    parser.add_argument('--gradient_accumulation_steps',type=int,  default=1)
    parser.add_argument('--max_grad_norm',type=int,  default=5)

    args = parser.parse_args()
    return args


def train_fn_decoder(args,device,epoch,train_loader, model, optimizer):
    kl_beta = args.kl_beta
    grad_norm = -1
    gas = args.gradient_accumulation_steps

    batch_time = AverageMeter()
    losses = AverageMeter()
    kl_losses = AverageMeter()
    clip_losses = AverageMeter()
    clip_losses_a = AverageMeter()
    clip_losses_f = AverageMeter()
    lm_losses = AverageMeter()
    gene_lm_losses = AverageMeter()

    end = time.time()
    accumulated_loss = 0

    N = len(train_loader)
    for curr_step, batch_data in enumerate(train_loader):
        
        inputs = batch_data[0].to(device)
        input_mask = batch_data[1].to(inputs.device)
        targets = batch_data[2].to(inputs.device)
        target_mask = batch_data[3].to(inputs.device)
        props = batch_data[4].to(inputs.device)
        frag_i = batch_data[-2].to(inputs.device)
        frag_t = batch_data[-1].to(inputs.device)
        batch_size = inputs.shape[0]

        output_dict = model(frag_i, frag_t, inputs, input_mask,  targets,target_mask, props, args.batch_size,if_gene=True) 
        a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8 = output_dict['logits_aug_a']
        f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8 = output_dict['logits_aug_f']
        logits_per_phar, logits_per_phar_2, logits_per_smiles, logits_per_smiles_2 = output_dict['logits']
        logits_per_phar_1_aug, logits_per_phar_2_aug, logits_per_smiles_1_aug, logits_per_smiles_2_aug = output_dict['logits_aug']
        kl = output_dict['z_klloss'] 
        lm_loss = output_dict['lm_loss'] 
        gene_lm_loss = output_dict['gene_lm_loss'] 

        # loss
        criterion = ClipInfoCELoss()

        clip_loss_1, target = criterion(logits_per_phar, logits_per_smiles)
        clip_loss_2, _ = criterion(logits_per_phar_2, logits_per_smiles_2)
        clip_loss_1_aug, _ = criterion(logits_per_phar_1_aug, logits_per_smiles_1_aug)
        clip_loss_2_aug, _ = criterion(logits_per_phar_2_aug, logits_per_smiles_2_aug)

        a_clip_loss_1, target_a = criterion(a_5, a_7)
        a_clip_loss_2, _ = criterion(a_6, a_8)
        a_clip_loss_1_aug, _ = criterion(a_1, a_3)
        a_clip_loss_2_aug, _ = criterion(a_2, a_4)

        f_clip_loss_1, target_f = criterion(f_5, f_7)
        f_clip_loss_2, _ = criterion(f_6, f_8)
        f_clip_loss_1_aug, _ = criterion(f_1, f_3)
        f_clip_loss_2_aug, _ = criterion(f_2, f_4)


        clip_loss = (clip_loss_1 + clip_loss_2 + clip_loss_1_aug + clip_loss_2_aug) / 4 
        clip_loss_a = (a_clip_loss_1 + a_clip_loss_2 + a_clip_loss_1_aug + a_clip_loss_2_aug) / 4 
        clip_loss_f = (f_clip_loss_1 + f_clip_loss_2 + f_clip_loss_1_aug + f_clip_loss_2_aug) / 4 
        loss = (clip_loss +clip_loss_a+clip_loss_f)/3 + lm_loss + kl*kl_beta + gene_lm_loss

        accumulated_loss += loss
        losses.update(loss.item(), batch_size)
        clip_losses.update(clip_loss.item(), batch_size)
        clip_losses_a.update(clip_loss_a.item(), batch_size)
        clip_losses_f.update(clip_loss_f.item(), batch_size)
        lm_losses.update(lm_loss.item(),batch_size)
        gene_lm_losses.update(gene_lm_loss.item(),batch_size)
        kl_losses.update(kl.item(), batch_size)

        if (curr_step + 1) % gas == 0:
            accumulated_loss = accumulated_loss / gas
            accumulated_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            accumulated_loss = 0
            
        # measure accuracy and record loss
        prec1, prec5 = accuracy(
            logits_per_phar, target, topk=(1, args.topk))
        prec1_a, prec5_a = accuracy(
            a_5, target_a, topk=(1, args.topk))
        prec1_f, prec5_f = accuracy(
            f_5, target_f, topk=(1, args.topk))
        
        batch_time.update(time.time() - end)
        end = time.time()
        from datetime import datetime
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y年%m月%d日 %H:%M:%S")
        if int(curr_step)%30 == 0:
            print(f'Epoch: [{epoch + 1}][{curr_step}/{len(train_loader)}], '
                f'{formatted_time}, '
                f"Train KL Loss: {kl_losses.val:.4f}({kl_losses.avg:.4f}), "
                f"Train LM Loss: {lm_losses.val:.4f}({lm_losses.avg:.4f}), "
                f"Train CLM Loss: {gene_lm_losses.val:.4f}({gene_lm_losses.avg:.4f}), "
                f"g_prec1: {prec1.item():.4f}, "
                f"g_prec5: {prec5.item():.4f}, "
                f"g_Clip Loss: {clip_losses.val:.4f}({clip_losses.avg:.4f}), "
                f"a_prec1: {prec1_a.item():.4f}, "
                f"a_prec5: {prec5_a.item():.4f}, "
                f"a_Clip Loss: {clip_losses_a.val:.4f}({clip_losses_a.avg:.4f}), "
                f"f_prec1: {prec1_f.item():.4f}, "
                f"f_prec5: {prec5_f.item():.4f}, "
                f"f_Clip Loss: {clip_losses_f.val:.4f}({clip_losses_f.avg:.4f}), "
                f'Grad: {grad_norm:.4f} ', flush=True)

    torch.cuda.empty_cache()
    return

def valid_fn_decoder(args,device,epoch,dataloader, model, dataclass = "valid"):

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    kl_losses = AverageMeter()
    clip_losses = AverageMeter()
    clip_losses_a = AverageMeter()
    clip_losses_f = AverageMeter()
    lm_losses = AverageMeter()
    cvae_lm_losses = AverageMeter()

    with torch.no_grad():
        for curr_step, batch_data in enumerate(dataloader):

            data_time.update(time.time() - end)
            inputs = batch_data[0].to(device)
            input_mask = batch_data[1].to(inputs.device)
            targets = batch_data[2].to(inputs.device)
            target_mask = batch_data[3].to(inputs.device)
            props = batch_data[4].to(inputs.device)
            frag_i = batch_data[-2].to(inputs.device)
            frag_t = batch_data[-1].to(inputs.device)

            batch_size = inputs.shape[0]
            output_dict = model(frag_i,frag_t,inputs, input_mask,  targets,target_mask, props,args.batch_size,if_gene=True) # 自编码器模型，输入前五个特征+后3个特征

            a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8 = output_dict['logits_aug_a']
            f_1,f_2,f_3,f_4,f_5,f_6,f_7,f_8 = output_dict['logits_aug_f']
            logits_per_phar, logits_per_phar_2, logits_per_smiles, logits_per_smiles_2 = output_dict['logits']
            logits_per_phar_1_aug, logits_per_phar_2_aug, logits_per_smiles_1_aug, logits_per_smiles_2_aug = output_dict['logits_aug']
            kl = output_dict['z_klloss'] 
            lm_loss = output_dict['lm_loss'] 
            cvae_lm_loss = output_dict['gene_lm_loss'] 

            criterion = ClipInfoCELoss()

            clip_loss_1, target = criterion(logits_per_phar, logits_per_smiles)
            clip_loss_2, _ = criterion(logits_per_phar_2, logits_per_smiles_2)
            clip_loss_1_aug, _ = criterion(logits_per_phar_1_aug, logits_per_smiles_1_aug)
            clip_loss_2_aug, _ = criterion(logits_per_phar_2_aug, logits_per_smiles_2_aug)

            a_clip_loss_1, target_a = criterion(a_5, a_7)
            a_clip_loss_2, _ = criterion(a_6, a_8)
            a_clip_loss_1_aug, _ = criterion(a_1, a_3)
            a_clip_loss_2_aug, _ = criterion(a_2, a_4)

            f_clip_loss_1, target_f = criterion(f_5, f_7)
            f_clip_loss_2, _ = criterion(f_6, f_8)
            f_clip_loss_1_aug, _ = criterion(f_1, f_3)
            f_clip_loss_2_aug, _ = criterion(f_2, f_4)
            
            clip_loss = (clip_loss_1 + clip_loss_2 + clip_loss_1_aug + clip_loss_2_aug) / 4 
            clip_loss_a = (a_clip_loss_1 + a_clip_loss_2 + a_clip_loss_1_aug + a_clip_loss_2_aug) / 4 
            clip_loss_f = (f_clip_loss_1 + f_clip_loss_2 + f_clip_loss_1_aug + f_clip_loss_2_aug) / 4 

            clip_losses.update(clip_loss.item(), batch_size)
            clip_losses_a.update(clip_loss_a.item(), batch_size)
            clip_losses_f.update(clip_loss_f.item(), batch_size)

            lm_losses.update(lm_loss.item(),batch_size)
            cvae_lm_losses.update(cvae_lm_loss.item(),batch_size)
            kl_losses.update(kl.item(), batch_size)

            prec1, prec5 = accuracy(
                logits_per_phar, target, topk=(1, args.topk))
            prec1_a, prec5_a = accuracy(
                a_5, target_a, topk=(1, args.topk))
            prec1_f, prec5_f = accuracy(
                f_5, target_f, topk=(1, args.topk))

            batch_time.update(time.time() - end)
            end = time.time()
            from datetime import datetime
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y年%m月%d日 %H:%M:%S")
            if int(curr_step)%10 == 0:
                if dataclass == "valid":
                    print(f'Epoch: [{epoch + 1}][{curr_step}/{len(dataloader)}], '
                        f'{formatted_time}, '
                        f"Valid KL Loss: {kl_losses.val:.4f}({kl_losses.avg:.4f}), "
                        f"Valid LM Loss: {lm_losses.val:.4f}({lm_losses.avg:.4f}), "
                        f"Valid CLM Loss: {cvae_lm_loss.val:.4f}({cvae_lm_loss.avg:.4f}), "
                        f"g_prec1: {prec1.item():.4f}, "
                        f"g_prec5: {prec5.item():.4f}, "
                        f"g_Clip Loss: {clip_losses.val:.4f}({clip_losses.avg:.4f}), "
                        f"a_prec1: {prec1_a.item():.4f}, "
                        f"a_prec5: {prec5_a.item():.4f}, "
                        f"a_Clip Loss: {clip_losses_a.val:.4f}({clip_losses_a.avg:.4f}), "
                        f"f_prec1: {prec1_f.item():.4f}, "
                        f"f_prec5: {prec5_f.item():.4f}, "
                        f"f_Clip Loss: {clip_losses_f.val:.4f}({clip_losses_f.avg:.4f}), ")
                else:
                    print(f'Epoch: [{epoch + 1}][{curr_step}/{len(dataloader)}], '
                        f'{formatted_time}, '
                        f"Test KL Loss: {kl_losses.val:.4f}({kl_losses.avg:.4f}), "
                        f"Test LM Loss: {lm_losses.val:.4f}({lm_losses.avg:.4f}), "
                        f"Test CLM Loss: {cvae_lm_loss.val:.4f}({cvae_lm_loss.avg:.4f}), "
                        f"g_prec1: {prec1.item():.4f}, "
                        f"g_prec5: {prec5.item():.4f}, "
                        f"g_Clip Loss: {clip_losses.val:.4f}({clip_losses.avg:.4f}), "
                        f"a_prec1: {prec1_a.item():.4f}, "
                        f"a_prec5: {prec5_a.item():.4f}, "
                        f"a_Clip Loss: {clip_losses_a.val:.4f}({clip_losses_a.avg:.4f}), "
                        f"f_prec1: {prec1_f.item():.4f}, "
                        f"f_prec5: {prec5_f.item():.4f}, "
                        f"f_Clip Loss: {clip_losses_f.val:.4f}({clip_losses_f.avg:.4f}), ")

    torch.cuda.empty_cache()
    return



def main():
    with open('./datasets/vocab.pickle', 'rb') as f:
        vocab = pickle.load(f)
    st_tokenizer = Tokenizer(Tokenizer.gen_vocabs(vocab)) # 建立的词汇表
    args = parse_args()
    device = args.ddevice

    # 3. models  
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
    state_dict = torch.load('./pretrained_models/VAE.pth')
    smiles_model = STransformer(model_params, st_tokenizer)
    new_state_dict = {}
    for key in state_dict:
        new_key = key.replace('module.', '')
        new_state_dict[new_key] = state_dict[key]
    smiles_model.load_state_dict(new_state_dict, strict=True)
    ps_model = PropModel(model_params,smiles_model)
    ps_model = ps_model.to(device)

    phargnn_optimizer = torch.optim.AdamW(ps_model.parameters(), lr=args.start_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(phargnn_optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1)

    ps_model.train()

    train_dataset = SemiSmilesDataset(args.exam_datapath, st_tokenizer,
                                      use_random_input_smiles = True, use_random_target_smiles = True, corrupt=True,normal_path=args.normal_path)
    valid_dataset = SemiSmilesDataset(args.va_datapath, st_tokenizer,
                                      use_random_input_smiles = True, use_random_target_smiles = True, corrupt=True,normal_path=args.normal_path)
    test_dataset = SemiSmilesDataset(args.te_datapath, st_tokenizer,
                                      use_random_input_smiles = True, use_random_target_smiles = True, corrupt=True,normal_path=args.normal_path)

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              sampler=train_sampler,
                              drop_last=True,
                              num_workers=args.num_workers, 
                              pin_memory=False, 
                              collate_fn=train_dataset.collate_fn,
                              persistent_workers=True) 
    valid_sampler = RandomSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=args.batch_size,
                              sampler=valid_sampler,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=valid_dataset.collate_fn,
                              persistent_workers=True)
    test_sampler = RandomSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                              batch_size=args.batch_size,
                              sampler=test_sampler,
                              num_workers=args.num_workers,
                              pin_memory=False,
                              drop_last=True,
                              collate_fn=test_dataset.collate_fn,
                              persistent_workers=True)
    
    for epoch in range(0,args.epochs):
        
        train_fn_decoder(args,device,epoch, train_loader, ps_model, phargnn_optimizer)
        if epoch % 10 == 0:
            torch.save(ps_model.state_dict(), './pretrained_models/'+str(epoch)+'.pth')
        valid_fn_decoder(args,device,epoch, valid_loader, ps_model)
        valid_fn_decoder(args,device,epoch, test_loader, ps_model,"Test")
        scheduler.step() 

if __name__ == '__main__':
    main()
