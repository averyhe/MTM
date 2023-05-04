max_threads=8
import os

os.environ["OMP_NUM_THREADS"] = f"{max_threads}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{max_threads}" 
os.environ["MKL_NUM_THREADS"] = f"{max_threads}" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{max_threads}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{max_threads}" 
import re

import torch

torch.set_num_threads(max_threads)

import argparse
import datetime

from tqdm import tqdm
from munch import Munch

import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from data import DataHub, ReadTissuePairDataset
from model import MTM
from utils import setup_exp_dir


class Trainer:
    def __init__(
        self,
        datahub,
        device: str,
        log_dir: str,
        lr: float = 5e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        batch_size: int = 256,
        n_workers: int = 12,
        max_epochs: int = 200,
        lambdas: dict = {
            'adv': 1,
            'ind': 1,
            'rec': 1,
            'cyc': 1,
        },
    ):
        self.datahub = datahub
        self.device = device
        self.log_dir = log_dir
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.max_epochs = max_epochs
        self.lambdas = lambdas
        
        self.start_epoch = 0
        
        train_indices, val_indices = self.datahub.split_item_indices()
        train_set = Subset(self.datahub.dataset, train_indices)
        val_set = Subset(self.datahub.dataset, val_indices)
        self.train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_workers,
            persistent_workers=True,
        )
        self.val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_workers,
            persistent_workers=True,
        )

        # * save individual ids for training and evaluation
        train_sample_info = self.datahub.dataset.data['sample']['info'].query(
            f'SMTSD in @self.datahub.dataset.train_tissues and Subject_id in @self.datahub.train_indivs'
        )
        train_indivs = list(train_sample_info.Subject_id.value_counts()[train_sample_info.Subject_id.value_counts()>1].index)
        val_sample_info = self.datahub.dataset.data['sample']['info'].query(
            f'SMTSD in @self.datahub.dataset.train_tissues and Subject_id in @self.datahub.val_indivs'
        )
        val_indivs = list(val_sample_info.Subject_id.value_counts()[val_sample_info.Subject_id.value_counts()>1].index)
        os.makedirs(os.path.join(self.log_dir, "data_split"))
        pd.Series(train_indivs).to_csv(os.path.join(self.log_dir, "data_split", "train_indivs.txt"), index=False, header=False)
        pd.Series(val_indivs).to_csv(os.path.join(self.log_dir, "data_split", "val_indivs.txt"), index=False, header=False)
        
        # * init model
        n_genes = len(self.datahub.genes_to_use)
        n_tissues = len(self.datahub.dataset.item2idx["tissue"])
        n_indiv = len(self.datahub.dataset.item2idx["indiv"])
        dim_code = 256
        dim_z = 32
        dropout = 0.3
        self.n_tissues = n_tissues
        self.dim_z = dim_z

        self.model = MTM(
            n_genes=n_genes,
            n_tissues=n_tissues,
            dim_code=dim_code,
            dim_z=dim_z,
            dropout=dropout,
            device=self.device,
        )
        
        self.model = self.model.to(self.device)
    
        self.model.setup_optimizers(lr=self.lr, b1=self.b1, b2=self.b2)
        
        self.writer = SummaryWriter(os.path.join(self.log_dir, "tb_logs"))

    def run(self):
        for epoch in tqdm(range(self.start_epoch, self.max_epochs+1)):
            self.epoch = epoch

            self.model.train_epoch(
                epoch, self.train_loader, self.datahub, self.writer,
                lambda_adv=self.lambdas['adv'], lambda_ind=self.lambdas['ind'], lambda_rec=self.lambdas['rec'], lambda_cyc=self.lambdas['cyc'],
            )

        self.model.save_ckpt(epoch, os.path.join(self.log_dir, "models"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--input_dir', type=str, help='the input directory')
    parser.add_argument('--expr', type=str, help='the expression file for model training in the input dir')
    parser.add_argument('--sample_attr', type=str, help='the sample attribute file for model training in the input dir')
    parser.add_argument('--gene_id', type=str, help='the gene id file for model training in the input dir')
    parser.add_argument('--indiv_id', type=str, help='the individual id file for model training in the input dir')
    parser.add_argument('--tissue_type', type=str, help='the tissue type file for model training in the input dir')
    parser.add_argument('--device', type=str, default="cuda:0", help='device to use')
    parser.add_argument('--output_dir', type=str, help='the output directory')

    parser.add_argument('--max_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=5e-4, help='Adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: b1')
    parser.add_argument('--b2', type=float, default=0.9, help='Adam: b2')
    parser.add_argument('--n_workers', type=int, default=4, help='number of cpu threads to use during batch generation')

    opt = parser.parse_args()

    device = torch.device(opt.device)
    args = Munch(
        expr_file=os.path.join(opt.input_dir, opt.expr),
        sample_attribute_file=os.path.join(opt.input_dir, opt.sample_attr),
        train_indiv_file=os.path.join(opt.input_dir, opt.indiv_id),
        train_tissue_file=os.path.join(opt.input_dir, opt.tissue_type),
        gene_list_file=os.path.join(opt.input_dir, opt.gene_id),
        tissue_labelname="SMTSD",
        train_tissue_source="Any",
        train_tissue_target="Any",
    )

    ds = ReadTissuePairDataset(
        train_indiv_file=args.train_indiv_file,
        train_tissue_file=args.train_tissue_file,
        expr_file=args.expr_file,
        sample_attribute_file=args.sample_attribute_file,
        tissue_labelname=args.tissue_labelname,
        tissue_source=args.train_tissue_source,
        tissue_target=args.train_tissue_target,
    )
    dh = DataHub(
        expr_file=args.expr_file,
        dataset=ds,
        gene_list_file=args.gene_list_file,
        train_ratio=None,
        cross_validation=True,
        current_fold=0,
    )
    dh.setup()
    print("Initialize training ...")

    rand_seed = dh.current_fold # ! not necessary
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_log_dir = setup_exp_dir(opt.output_dir, suffix=f"MTM")

    trainer = Trainer(
        datahub=dh,
        device=device,
        log_dir=exp_log_dir,
        lr=opt.lr,
        b1=opt.b1,
        b2=opt.b2,
        batch_size=opt.batch_size,
        max_epochs=opt.max_epochs,
        n_workers=opt.n_workers,
        lambdas={
            'adv': 1.0,
            'ind': 1.0,
            'rec': 1.0,
            'cyc': 1.0,
        }
    )
    print("Training ...")
    trainer.run()
