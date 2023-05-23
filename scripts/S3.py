max_threads=8
import os

os.environ["OMP_NUM_THREADS"] = f"{max_threads}" 
os.environ["OPENBLAS_NUM_THREADS"] = f"{max_threads}" 
os.environ["MKL_NUM_THREADS"] = f"{max_threads}" 
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{max_threads}" 
os.environ["NUMEXPR_NUM_THREADS"] = f"{max_threads}" 
import re

import torch
import torch.nn as nn

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
from utils import setup_exp_dir



class S3(nn.Module):
    def __init__(
        self,
        n_genes: int,
        dim_code: int = 256,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_genes = n_genes
        self.dim_code = dim_code
        self.device = device

        self.model_E = S3_encoder(
            n_genes=n_genes,
            dim_code=dim_code,
        )
        self.model_G = S3_decoder(
            n_genes=n_genes,
            dim_code=dim_code,
        )

        self.criterion = nn.L1Loss()
    
    def setup_optimizers(
        self,
        lr: float=5e-4,
        b1: float=0.5,
        b2: float=0.9,
    ):
        self.opt_E = torch.optim.Adam(
            [{"params": self.model_E.parameters()}], lr, [b1, b2]
        )
        self.opt_G = torch.optim.Adam(
            [{"params": self.model_G.parameters()}], lr, [b1, b2]
        )

    def save_ckpt(
        self,
        epoch: int,
        ckpt_path: str,
    ):
        ckpt = {
            "epoch": epoch,
            "model_E_params": self.model_E.state_dict(),
            "model_G_params": self.model_G.state_dict(),
            "opt_E": self.opt_E.state_dict(),
            "opt_G": self.opt_G.state_dict(),
        }
        print("Saving checkpoint to {}".format(ckpt_path))
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path, exist_ok=True)
        torch.save(ckpt, os.path.join(ckpt_path, "model_ckpt.tar"))
    
    def load_ckpt(
        self,
        ckpt_path: str,
    ):
        print("Loading checkpoint from {}".format(ckpt_path))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.start_epoch = ckpt["epoch"]

        self.model_E.load_state_dict(ckpt["model_E_params"])
        self.model_G.load_state_dict(ckpt["model_G_params"])

        self.opt_E.load_state_dict(ckpt["opt_E"])
        self.opt_G.load_state_dict(ckpt["opt_G"])

        self.model_E.to(self.device)
        self.model_G.to(self.device)

    def train_epoch(
        self,
        epoch: int,
        data_loader,
        datahub,
        writer=None,
        lambda_ind=2,
    ):
        self.model_E.train()
        self.model_G.train()

        loss_eg = 0
        loss_ind = 0

        for batch_idx, batch in enumerate(data_loader):
            losses = self.train_step(
                batch, datahub, lambda_ind,
            )
            loss_eg += losses["loss_eg"]
            loss_ind += losses["loss_ind"]
        
        loss_eg /= len(data_loader)
        loss_ind /= len(data_loader)

        if writer is not None:
            writer.add_scalar(
                "loss_eg", loss_eg, epoch
            )
            writer.add_scalar(
                "loss_ind", loss_ind, epoch
            )

        # print(
        #     "|Epoch: {:5d} | loss_d: {:10.2e} | loss_adv: {:10.2e} | loss_ind: {:10.2e} | loss_rec: {:10.2e} | loss_cyc: {:10.2e} |"
        #     .format(epoch, loss_d, loss_adv, loss_ind, loss_rec, loss_cyc)
        # )
        # return

    def train_step(
        self,
        batch,
        datahub,
        lambda_ind=2,
    ):
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

        code_s = self.model_E(expr_s)
        expr_t_tra = self.model_G(code_s)
        
        loss_ind = self.criterion(expr_t_tra, expr_t)

        loss_eg = lambda_ind*loss_ind
        self.opt_E.zero_grad()
        self.opt_G.zero_grad()
        loss_eg.backward()
        self.opt_E.step()
        self.opt_G.step()

        losses = {
            "loss_eg": loss_eg,
            "loss_ind": loss_ind,
        }
        return losses
    
    def predict(
        self,
        expr_tpm: pd.DataFrame,
        tissue_source: str,
        tissue_target: str,
        datahub,
    ):
        '''
            expr_tpm: pd.DataFrame, shape (n_samples, n_genes)
            please make sure batch effect is removed
        '''
        self.model_E.eval()
        self.model_G.eval()

        expr_tpm_s = torch.tensor(expr_tpm.values, dtype=torch.float32).to(self.device)
        tissue_s = datahub.dataset.item2idx["tissue"][tissue_source]
        tissue_t = datahub.dataset.item2idx["tissue"][tissue_target]
        
        # repeat batch size
        tissue_s = torch.tensor([tissue_s] * expr_s.shape[0]).to(self.device)
        tissue_t = torch.tensor([tissue_t] * expr_s.shape[0]).to(self.device)

        expr_s = datahub.tensorized_transform(expr_tpm_s, tissue_s)

        code = self.model_E(expr_s)
        expr_t_fake = self.model_G(code)
        expr_t_fake_untransformed = datahub.tensorized_untransform(expr_t_fake, tissue_t)
        expr_tpm_pred = pd.DataFrame(
            expr_t_fake_untransformed.detach().cpu().numpy(),
            index=[i + f"_{tissue_target}_pred" for i in expr_tpm.index],
            columns=expr_tpm.columns,
        )
        return expr_tpm_pred


class S3_encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        dim_code: int,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.dim_code = dim_code
        
        self.mid_act = nn.LeakyReLU()
        self.layers = nn.ModuleList([
            nn.Linear(self.n_genes, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 512),
        ])
        self.layer_out = nn.Linear(512, self.dim_code)
    
    def forward(self, expr: torch.Tensor):
        h = expr
        for layer in self.layers:
            h = layer(h)
            h = self.mid_act(h)
        code = self.layer_out(h)
        return code


class S3_decoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        dim_code: int,
    ):
        super().__init__()
        self.dim_code = dim_code
        self.n_genes = n_genes
        
        self.mid_act = nn.LeakyReLU()
        self.layers = nn.ModuleList([
            nn.Linear(self.dim_code, 1024),
            nn.Linear(1024, 1024),
            nn.Linear(1024, 2048),
        ])
        self.layer_out = nn.Linear(2048, self.n_genes)
    
    def forward(self, code: torch.Tensor, output_acts: bool = False):
        h = code
        for layer in self.layers:
            h = layer(h)
            h = self.mid_act(h)
        expr_hat = self.layer_out(h)
        if output_acts:
            return expr_hat, h
        return expr_hat


class Trainer_S3:
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
        max_epochs: int = 1000,
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
        
        # * init model
        n_genes = len(self.datahub.genes_to_use)
        n_tissues = len(self.datahub.dataset.item2idx["tissue"])
        n_indiv = len(self.datahub.dataset.item2idx["indiv"])
        dim_code = 256

        self.model = S3(
            n_genes=n_genes,
            dim_code=dim_code,
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

    parser.add_argument('--max_epochs', type=int, default=1000, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=5e-4, help='Adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='Adam: b1')
    parser.add_argument('--b2', type=float, default=0.9, help='Adam: b2')
    parser.add_argument('--n_workers', type=int, default=4, help='number of cpu threads to use during batch generation')

    opt = parser.parse_args()

    device = torch.device(opt.device)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d")

    tissue_list = sorted(pd.read_csv(opt.tissue_type, sep='\t', dtype='str', header=None).iloc[:, 0].tolist())
    for tissue in tissue_list:
        args = Munch(
            expr_file=os.path.join(opt.input_dir, opt.expr),
            sample_attribute_file=os.path.join(opt.input_dir, opt.sample_attr),
            train_indiv_file=os.path.join(opt.input_dir, opt.indiv_id),
            train_tissue_file=os.path.join(opt.input_dir, opt.tissue_type),
            gene_list_file=os.path.join(opt.input_dir, opt.gene_id),
            tissue_labelname="SMTSD",
            train_tissue_source="Whole_Blood",
            train_tissue_target=tissue,
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

        exp_log_dir = setup_exp_dir(
            os.path.join(opt.output_dir, "S3"),
            prefix=tissue,
        )

        trainer = Trainer_S3(
            datahub=dh,
            device=device,
            log_dir=exp_log_dir,
            lr=opt.lr,
            b1=opt.b1,
            b2=opt.b2,
            batch_size=opt.batch_size,
            max_epochs=opt.max_epochs,
            n_workers=opt.n_workers,
        )
        print("Training ...")
        trainer.run()
