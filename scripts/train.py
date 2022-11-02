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

import numpy as np
import pandas as pd
import torch.nn.functional as F
from dataprocess import DataHub
from dataset import ReadTissuePairDataset
from munch import Munch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from model import *


class MTM:
    def __init__(
        self,
        datahub,
        device: str,
        log_dir: str,
        lr_D: float = 5e-4,
        lr_G: float = 5e-4,
        b1: float = 0.5,
        b2: float = 0.9,
        batch_size: int = 256,
        n_workers: int = 12,
        max_epochs: int = 200,
    ):
        self.datahub = datahub
        self.device = device
        self.log_dir = log_dir
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.b1 = b1
        self.b2 = b2
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.max_epochs = max_epochs
        
        self.start_epoch = 0
        
        self.datahub.setup()
        
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

        
        n_genes = len(self.datahub.genes_to_use)
        n_tissues = len(self.datahub.dataset.item2idx["tissue"])
        n_indiv = len(self.datahub.dataset.item2idx["indiv"])
        dim_code = 256
        dim_z = 32
        dropout = 0.3
        self.n_tissues = n_tissues
        self.dim_z = dim_z
        self.lr_E = self.lr_G

        self.model_E = Encoder(
            n_genes=n_genes,
            n_tissues=n_tissues,
            dim_code=dim_code,
            dropout=dropout,
        )
        self.model_G = Generator(
            n_genes=n_genes,
            n_tissues=n_tissues,
            dim_code=dim_code,
            dropout=dropout,
        )
        self.model_D = Discriminator(
            n_genes=n_genes,
            n_tissues=n_tissues,
        )
        self.model_M = Mapping(
            dim_z=self.dim_z,
            dim_code=dim_code,
            n_tissues=n_tissues,
            dropout=dropout,
        )

        self.model_E = self.model_E.to(self.device)
        self.model_G = self.model_G.to(self.device)
        self.model_D = self.model_D.to(self.device)
        self.model_M = self.model_M.to(self.device)
    
        self.opt_d = torch.optim.Adam(
            [{"params": self.model_D.parameters()}], self.lr_D, [self.b1, self.b2]
        )
        self.opt_g = torch.optim.Adam(
            [{"params": self.model_G.parameters()}], self.lr_G, [self.b1, self.b2]
        )
        self.opt_e = torch.optim.Adam(
            [{"params": self.model_E.parameters()}], self.lr_E, [self.b1, self.b2]
        )
        self.opt_m = torch.optim.Adam(
            self.model_M.parameters(), self.lr_E, [self.b1, self.b2]
        )
        
        self.criterion_l1 = nn.L1Loss()
        self.criterion_l2 = nn.MSELoss(reduction='mean')
        
        self.writer = SummaryWriter(os.path.join(self.log_dir, "tb_logs"))
    
    def run(self):
        for epoch in range(self.start_epoch, self.max_epochs+1):
            self.epoch = epoch
            
            # * Train
            self.model_E.train()
            self.model_G.train()
            self.model_D.train()
            self.model_M.train()

            for batch_idx, batch in enumerate(self.train_loader):

                training_steps = self.epoch * len(self.train_loader) + batch_idx
                to_unpack = [
                    batch["indiv_id"],
                    batch["tissue_s"],
                    batch["tissue_t"],
                    batch["expr_s"],
                    batch["expr_t"],
                ]
                (indiv_id, tissue_s, tissue_t, expr_s, expr_t) = [
                    item.to(self.device) for item in to_unpack
                ]
                batch_size = expr_s.shape[0]

                expr_tpm_s = self.datahub.tensorized_unnorm(expr_s, tissue_s)
                expr_tpm_t = self.datahub.tensorized_unnorm(expr_t, tissue_t)

                code_s = self.model_E(expr_tpm_s, tissue_s)
                expr_t_tra = self.model_G(code_s, tissue_t)
                z = torch.randn(batch_size, self.dim_z).to(self.device)
                z_codes = self.model_M(z)
                tissue_rand = torch.randint(
                    low=0, high=self.n_tissues, size=[batch_size]
                ).to(self.device)
                expr_rand = self.model_G(z_codes, tissue_rand)
                expr_tpm_rand = self.datahub.tensorized_unnorm(
                    expr_rand, tissue_rand
                )
                ez_codes = self.model_E(expr_tpm_rand, tissue_rand)
                expr_tpm_t_tra = self.datahub.tensorized_unnorm(
                    expr_t_tra, tissue_t
                )
                code_cyc_t = self.model_E(expr_tpm_t_tra, tissue_t)
                expr_cyc_s = self.model_G(code_cyc_t, tissue_s)

                # * train D
                score_real_s = self.model_D(expr_s, tissue_s)
                score_real_t = self.model_D(expr_t, tissue_t)
                loss_d_real = (F.relu(1 - score_real_t)).mean() + (
                    F.relu(1 - score_real_s)
                ).mean()
                score_fake_rand = self.model_D(expr_rand.detach(), tissue_rand)
                loss_d_fake = (F.relu(1 + score_fake_rand)).mean()

                loss_d = loss_d_real + loss_d_fake
                self.opt_d.zero_grad()
                loss_d.backward()
                self.opt_d.step()

                self.writer.add_scalar("d_loss", loss_d, training_steps)

                # * train E,M,G
                loss_gm_adv = -(self.model_D(expr_rand, tissue_rand).mean())
                loss_eg_reg = self.criterion_l1(expr_t_tra, expr_t)
                loss_em_rec_code = self.criterion_l1(z_codes, ez_codes)
                loss_eg_cyc = self.criterion_l1(expr_cyc_s, expr_s)

                loss_emg = loss_gm_adv + loss_eg_reg + loss_em_rec_code + loss_eg_cyc
                self.opt_e.zero_grad()
                self.opt_m.zero_grad()
                self.opt_g.zero_grad()
                loss_emg.backward()
                self.opt_e.step()
                self.opt_m.step()
                self.opt_g.step()

                self.writer.add_scalar(
                    "cyc_expr", loss_eg_cyc, training_steps
                )
                self.writer.add_scalar(
                    "l1", loss_eg_reg, training_steps
                )
                self.writer.add_scalar(
                    "cyc_code", loss_em_rec_code, training_steps
                )
                
            print(
                "|Epoch: {:5d} | loss_d: {:10.2e} | loss_gm_adv: {:10.2e} | loss_eg_reg: {:10.2e} | loss_em_rec_code: {:10.2e} | loss_eg_cyc: {:10.2e} |"
                .format(epoch, loss_d, loss_gm_adv, loss_eg_reg, loss_em_rec_code, loss_eg_cyc)
            )

