import torch
import torch.nn as nn
import torch.nn.functional as F

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

        self.mid_act = nn.LeakyReLU()
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
            h = self.mid_act(h)
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

        self.mid_act = nn.LeakyReLU()
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
            h = self.mid_act(h)
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
        self.mid_act = nn.LeakyReLU()
        
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
            h = self.mid_act(h)
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

    def forward(self, z: torch.Tensor, tissues=None) -> torch.Tensor:
        h = z
        for L in self.Ls:
            h = L(h)
            h = self.mid_act(h)
            if self.dropout:
                h = self.dropout_layer(h)
        latent_codes = self.layer_out(h)
        return latent_codes

