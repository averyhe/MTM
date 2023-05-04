import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from model_functions import TCM

# ------ models ------ #


class Encoder(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_tissues: int,
        dim_code: int,
        dropout: float = None,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.dim_code = dim_code
        self.dropout = dropout

        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList(
            [
                TCM(self.n_genes, 2048, self.n_tissues),
                TCM(2048, 1024, self.n_tissues),
                TCM(1024, 512, self.n_tissues),
            ]
        )
        self.layer_out = nn.Linear(512, self.dim_code)

    def forward(self, expr: torch.Tensor, tissues: int):
        expr = nn.functional.relu(expr)
        expr = torch.log2(expr + 1)
        h = expr
        for layer in self.layers:
            h = layer(h, tissues)
            if self.dropout is not None:
                h = self.dropout_layer(h)
        code = self.layer_out(h)
        return code


class Generator(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_tissues: int,
        dim_code: int,
        dropout: float = None,
    ):
        super().__init__()
        self.dim_code = dim_code
        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.dropout = dropout

        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)
        self.layers = nn.ModuleList(
            [
                TCM(self.dim_code, 1024, self.n_tissues),
                TCM(1024, 1024, self.n_tissues),
                TCM(1024, 2048, self.n_tissues),
            ]
        )
        self.layer_out = nn.Linear(2048, self.n_genes)

    def forward(
        self,
        code: torch.Tensor,
        tissues: int,
    ):
        h = code
        for layer in self.layers:
            h = layer(h, tissues)
            if self.dropout is not None:
                h = self.dropout_layer(h)
        expr_hat = self.layer_out(h)
        return expr_hat


class Discriminator(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_tissues: int,
        dim_emb: int = 100,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.dim_emb = dim_emb
        
        self.layers = nn.ModuleList(
            [
                nn.utils.spectral_norm(TCM(self.n_genes, 2048, self.n_tissues)),
                nn.utils.spectral_norm(TCM(2048, 1024, self.n_tissues)),
                nn.utils.spectral_norm(TCM(1024, self.dim_emb, self.n_tissues)),
            ]
        )
        
        self.layer_out = nn.utils.spectral_norm(nn.Linear(self.dim_emb, 1))

    def forward(self, expr: torch.Tensor, tissues: int):
        h = expr
        for layer in self.layers:
            h = layer(h, tissues)
        score = self.layer_out(h)
        return score


class Mapping(nn.Module):
    def __init__(
        self,
        dim_z: int,
        dim_code: int,
        n_tissues: int,
        dropout=None,
    ) -> None:
        super().__init__()
        self.dim_z = dim_z
        self.dim_code = dim_code
        self.n_tissues = n_tissues
        self.dropout = dropout

        self.mid_act = nn.LeakyReLU()
        if self.dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)

        self.Ls = nn.ModuleList(
            [
                nn.Linear(self.dim_z, 64),
                nn.Linear(64, 128),
                nn.Linear(128, 256),
            ]
        )
        self.layer_out = nn.Linear(256, self.dim_code)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for L in self.Ls:
            h = L(h)
            h = self.mid_act(h)
            if self.dropout:
                h = self.dropout_layer(h)
        latent_codes = self.layer_out(h)
        return latent_codes


class MTM(nn.Module):
    def __init__(
        self,
        n_genes: int,
        n_tissues: int,
        dim_code: int = 256,
        dim_z: int = 32,
        dropout: float = 0.3,
        device: str = "cpu",
    ):
        super().__init__()
        self.n_genes = n_genes
        self.n_tissues = n_tissues
        self.dim_code = dim_code
        self.dim_z = dim_z
        self.device = device

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
            dim_z=dim_z,
            dim_code=dim_code,
            n_tissues=n_tissues,
            dropout=dropout,
        )

        self.criterion = nn.L1Loss()

    def setup_optimizers(
        self,
        lr: float=5e-4,
        b1: float=0.5,
        b2: float=0.9,
    ):
        self.opt_D = torch.optim.Adam(
            [{"params": self.model_D.parameters()}], lr, [b1, b2]
        )
        self.opt_G = torch.optim.Adam(
            [{"params": self.model_G.parameters()}], lr, [b1, b2]
        )
        self.opt_E = torch.optim.Adam(
            [{"params": self.model_E.parameters()}], lr, [b1, b2]
        )
        self.opt_M = torch.optim.Adam(
            self.model_M.parameters(), lr, [b1, b2]
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
            "model_D_params": self.model_D.state_dict(),
            "model_M_params": self.model_M.state_dict(),
            "opt_E": self.opt_E.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "opt_M": self.opt_M.state_dict(),
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
        self.model_D.load_state_dict(ckpt["model_D_params"])
        self.model_M.load_state_dict(ckpt["model_M_params"])

        self.opt_E.load_state_dict(ckpt["opt_E"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.opt_M.load_state_dict(ckpt["opt_M"])

        self.model_E.to(self.device)
        self.model_G.to(self.device)
        self.model_D.to(self.device)
        self.model_M.to(self.device)

    def train_epoch(
        self,
        epoch: int,
        data_loader,
        datahub,
        writer=None,
        lambda_adv=1,
        lambda_ind=1,
        lambda_rec=1,
        lambda_cyc=1,
    ):
        self.model_E.train()
        self.model_G.train()
        self.model_D.train()
        self.model_M.train()

        loss_d = 0
        loss_adv = 0
        loss_ind = 0
        loss_rec = 0
        loss_cyc = 0

        for batch_idx, batch in enumerate(data_loader):
            losses = self.train_step(
                batch, datahub, lambda_adv, lambda_ind, lambda_rec, lambda_cyc,
            )
            loss_d += losses["loss_d"]
            loss_adv += losses["loss_adv"]
            loss_ind += losses["loss_ind"]
            loss_rec += losses["loss_rec"]
            loss_cyc += losses["loss_cyc"]
        
        loss_d /= len(data_loader)
        loss_adv /= len(data_loader)
        loss_ind /= len(data_loader)
        loss_rec /= len(data_loader)
        loss_cyc /= len(data_loader)

        if writer is not None:
            writer.add_scalar("loss_d", loss_d, epoch)
            writer.add_scalar(
                "loss_cyc", loss_cyc, epoch
            )
            writer.add_scalar(
                "loss_ind", loss_ind, epoch
            )
            writer.add_scalar(
                "loss_rec", loss_rec, epoch
            )
            writer.add_scalar(
                "loss_g", loss_adv, epoch
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
        lambda_adv=1,
        lambda_ind=1,
        lambda_rec=1,
        lambda_cyc=1,
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

        expr_tpm_s = datahub.tensorized_untransform(expr_s, tissue_s)

        code_s = self.model_E(expr_tpm_s, tissue_s)
        expr_t_tra = self.model_G(code_s, tissue_t)
        z = torch.randn(batch_size, self.dim_z).to(self.device)
        z_codes = self.model_M(z)
        tissue_rand = torch.randint(
            low=0, high=self.n_tissues, size=[batch_size]
        ).to(self.device)
        expr_rand = self.model_G(z_codes, tissue_rand)
        expr_tpm_rand = datahub.tensorized_untransform(
            expr_rand, tissue_rand
        )
        ez_codes = self.model_E(expr_tpm_rand, tissue_rand)
        expr_tpm_t_tra = datahub.tensorized_untransform(
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

        loss_d = lambda_adv * (loss_d_real + loss_d_fake)
        self.opt_D.zero_grad()
        loss_d.backward()
        self.opt_D.step()

        # * train E,M,G
        loss_adv = -(self.model_D(expr_rand, tissue_rand).mean())
        loss_ind = self.criterion(expr_t_tra, expr_t)
        loss_rec = self.criterion(z_codes, ez_codes)
        loss_cyc = self.criterion(expr_cyc_s, expr_s)

        loss_emg = lambda_adv*loss_adv + lambda_ind*loss_ind + lambda_rec*loss_rec + lambda_cyc*loss_cyc
        self.opt_E.zero_grad()
        self.opt_M.zero_grad()
        self.opt_G.zero_grad()
        loss_emg.backward()
        self.opt_E.step()
        self.opt_M.step()
        self.opt_G.step()

        losses = {
            "loss_d": loss_d,
            "loss_emg": loss_emg,
            "loss_adv": loss_adv,
            "loss_ind": loss_ind,
            "loss_rec": loss_rec,
            "loss_cyc": loss_cyc,
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
        self.model_D.eval()
        self.model_M.eval()

        expr_s = torch.tensor(expr_tpm.values, dtype=torch.float32).to(self.device)
        tissue_s = datahub.dataset.item2idx["tissue"][tissue_source]
        tissue_t = datahub.dataset.item2idx["tissue"][tissue_target]
        # repeat batch size
        tissue_s = torch.tensor([tissue_s] * expr_s.shape[0]).to(self.device)
        tissue_t = torch.tensor([tissue_t] * expr_s.shape[0]).to(self.device)

        code = self.model_E(expr_s, tissue_s)
        expr_t_fake = self.model_G(code, tissue_t)
        expr_t_fake_untransformed = datahub.tensorized_untransform(expr_t_fake, tissue_t)
        expr_tpm_pred = pd.DataFrame(
            expr_t_fake_untransformed.detach().cpu().numpy(),
            index=[i + f"_{tissue_target}_pred" for i in expr_tpm.index],
            columns=expr_tpm.columns,
        )
        return expr_tpm_pred

    def get_inferenced_data(
        self,
        epoch: int,
        data_loader,
        datahub,
        tissue_source: str = "Whole_Blood",
        tissue_target: str = "Any",
    ):
        assert (
            tissue_source in ["Any"] + list(datahub.dataset.item2idx["tissue"].keys())
        ) and (
            tissue_target in ["Any"] + list(datahub.dataset.item2idx["tissue"].keys())
        ), "tissue_source and tissue_target must be in the list of tissues."
        
        self.model_E.eval()
        self.model_G.eval()
        self.model_D.eval()
        self.model_M.eval()

        # * Initialize
        expr_t_real_transformed_list = []
        expr_t_fake_transformed_list = []
        expr_t_real_untransformed_list = []
        expr_t_fake_untransformed_list = []
        expr_indiv_list = []
        expr_tissue_t_list = []
        expr_tissue_s_list = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                inf_results = self.get_inferenced_data_step(
                    batch, datahub,
                    tissue_source, tissue_target,
                )
                if inf_results is None:
                    continue

                # * save data
                expr_t_real_transformed_list.extend(inf_results["expr_t_real_transformed"])
                expr_t_fake_transformed_list.extend(inf_results["expr_t_fake_transformed"])
                expr_t_real_untransformed_list.extend(inf_results["expr_t_real_untransformed"])
                expr_t_fake_untransformed_list.extend(inf_results["expr_t_fake_untransformed"])
                expr_tissue_t_list.extend(inf_results["tissue_t"])
                expr_tissue_s_list.extend(inf_results["tissue_s"])
                expr_indiv_list.extend(inf_results["expr_indiv"])
        
        # * to numpy
        expr_t_real_transformed = np.stack(expr_t_real_transformed_list)
        expr_t_fake_transformed = np.stack(expr_t_fake_transformed_list)
        expr_t_real_untransformed = np.stack(expr_t_real_untransformed_list)
        expr_t_fake_untransformed = np.stack(expr_t_fake_untransformed_list)
        expr_tissue_t = np.array(expr_tissue_t_list)
        expr_tissue_s = np.array(expr_tissue_s_list)
        expr_indiv_array = np.array(expr_indiv_list)
        
        inf_results_numpy = {
            "expr_t_real_transformed": expr_t_real_transformed,
            "expr_t_fake_transformed": expr_t_fake_transformed,
            "expr_t_real_untransformed": expr_t_real_untransformed,
            "expr_t_fake_untransformed": expr_t_fake_untransformed,
            "expr_tissue_t": expr_tissue_t,
            "expr_tissue_s": expr_tissue_s,
            "expr_indiv": expr_indiv_array,
        }
        return inf_results_numpy

    def get_inferenced_data_step(
        self,
        batch,
        datahub,
        tissue_source,
        tissue_target,
    ):
        if tissue_source != "Any":
            tissue_source_idx = datahub.dataset.item2idx["tissue"][tissue_source]
        if tissue_target != "Any":
            tissue_target_idx = datahub.dataset.item2idx["tissue"][tissue_target]
        
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
        val_batch_size = expr_s.shape[0]

        # * mask
        if (tissue_source == "Any") and (tissue_target == "Any"):
            mask = torch.ones(val_batch_size, dtype=torch.bool, device=self.device)
        elif (tissue_source == "Any") and (tissue_target != "Any"):
            mask = tissue_t == tissue_target_idx
        elif (tissue_source != "Any") and (tissue_target == "Any"):
            mask = tissue_s == tissue_source_idx
        elif (tissue_source != "Any") and (tissue_target != "Any"):
            mask_s = tissue_s == tissue_source_idx
            mask_t = tissue_t == tissue_target_idx
            mask = mask_s * mask_t
        
        if mask.sum() == 0:
            return

        (indiv_id, tissue_s, tissue_t, expr_s, expr_t) = [
            item[mask] for item in (indiv_id, tissue_s, tissue_t, expr_s, expr_t)
        ]

        # * inference
        expr_tpm_s = datahub.tensorized_untransform(expr_s, tissue_s)
        code_s = self.model_E(expr_tpm_s, tissue_s)
        expr_t_fake = self.model_G(code_s, tissue_t)

        expr_t_real_untransformed = datahub.tensorized_untransform(expr_t, tissue_t)
        expr_t_fake_untransformed = datahub.tensorized_untransform(expr_t_fake, tissue_t)
        
        inf_results = {
            "tissue_s": tissue_s.cpu().numpy(),
            "tissue_t": tissue_t.cpu().numpy(),
            "expr_indiv": indiv_id.cpu().numpy(),
            "expr_t_real_transformed": expr_t.cpu().numpy(),
            "expr_t_fake_transformed": expr_t_fake.cpu().numpy(),
            "expr_t_real_untransformed": expr_t_real_untransformed.cpu().numpy(),
            "expr_t_fake_untransformed": expr_t_fake_untransformed.cpu().numpy(),
        }
        return inf_results