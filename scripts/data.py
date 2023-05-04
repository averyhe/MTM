import os
import json
import copy
import random
import pandas as pd
import numpy as np
import datatable as dt
import torch
from torch.utils.data import Dataset, DataLoader

from transform_functions import *


def read_bundle_data(
    expr_file,
    sample_attribute_file,
    tissue_labelname='SMTSD',
    indiv_labelname='Subject_id',
):
    # * read data
    data = {
        'sample': {},
        'indiv': {},
    }
    if expr_file.endswith('.txt'):
        expr_data = dt.fread(expr_file, sep="\t", header=True).to_pandas()
        expr_data = expr_data.set_index(expr_data.columns[0]).astype('float32')
    elif expr_file.endswith('.ptl'):
        expr_data = torch.load(expr_file)
    elif expr_file.endswith('.gz'):
        expr_data = pd.read_csv(
            expr_file, compression='gzip', sep='\t', converters={0: str}, dtype='float32', index_col=0
        )
    else:
        raise ValueError(f"Unknown file type: {expr_file}")
    
    data['expr'] = expr_data

    data['sample']['info'] = pd.read_csv(sample_attribute_file, sep='\t', index_col=0)
    data['indiv']['info'] = None

    # * index data
    index_data = {
        "sample_dict": {item: idx for idx, item in enumerate(sorted(list(set(data['sample']['info'].index))))},
        "indiv_dict": {item: idx for idx, item in enumerate(sorted(list(set(data['sample']['info'].loc[:, indiv_labelname]))))},
        "tissue_dict": {item: idx for idx, item in enumerate(sorted(list(set(data['sample']['info'].loc[:, tissue_labelname]))))},
    }
    item2idx = {
        'sample': index_data['sample_dict'],
        'indiv': index_data['indiv_dict'],
        'tissue': index_data['tissue_dict'],
    }
    idx2item = {
        'sample': {},
        'indiv': {},
        'tissue': {},
    }
    for key in index_data['sample_dict']:
        val = index_data['sample_dict'][key]
        idx2item['sample'][str(val)] = key
    for key in index_data['indiv_dict']:
        val = index_data['indiv_dict'][key]
        idx2item['indiv'][str(val)] = key
    for key in index_data['tissue_dict']:
        val = index_data['tissue_dict'][key]
        idx2item['tissue'][str(val)] = key

    # * correspondance between sample and indiv
    corr_data = pd.DataFrame({
        'sample_id': data['sample']['info'].index,
        'indiv_id': data['sample']['info'].loc[:, indiv_labelname],
    })
    sample2indiv = {corr_data.loc[:, 'sample_id'][i]: corr_data.loc[:, 'indiv_id'][i] for i in range(corr_data.shape[0])}
    
    sample2tissue = {}
    indiv2sample = {}
    sample_list = list(item2idx['sample'].keys())
    for sample_id in sample_list:
        indiv_id = sample2indiv[sample_id]
        tissue = data['sample']['info'].loc[sample_id, tissue_labelname]
        sample2tissue[sample_id] = tissue
        if indiv_id not in indiv2sample:
            indiv2sample[indiv_id] = {
                'tissues': [],
                'samples': [],
            }
        indiv2sample[indiv_id]['tissues'].append(tissue)
        indiv2sample[indiv_id]['samples'].append(sample_id)
        indiv2sample[indiv_id][tissue] = sample_id
    (
        data['sample']['sample2indiv'],
        data['sample']['sample2tissue'],
        data['indiv']['indiv2sample']
    ) = sample2indiv, sample2tissue, indiv2sample
    return data, item2idx, idx2item


class BaseDataset(Dataset):
    def __init__(
        self,
        expr_file,
        sample_attribute_file,
        tissue_labelname='SMTSD',
        indiv_labelname='Subject_id',
    ) -> None:
        super().__init__()
        self.data, self.item2idx, self.idx2item = read_bundle_data(
            expr_file,
            sample_attribute_file,
            tissue_labelname=tissue_labelname,
            indiv_labelname=indiv_labelname,
        )

class BuildTissuePairDataset(BaseDataset):
    def __init__(
        self,
        train_indiv_file,
        train_tissue_file,
        expr_file,
        sample_attribute_file,
        tissue_labelname='SMTSD',
        indiv_labelname='Subject_id',
    ):
        super().__init__(expr_file, sample_attribute_file, tissue_labelname, indiv_labelname)
        self.train_indivs, self.train_samples, self.train_tissues = self.filter_train(train_indiv_file, train_tissue_file)
        self.data_dict = self.build_data_dict()
    
    def build_data_dict(self, save_path=None):
        indiv_list = self.item2idx['indiv'].keys()
        D = {}
        for indiv_id in indiv_list:
            D[indiv_id] = {
                'indiv': {},
                'sample': {},
            }
            
            indiv_samples = self.data['indiv']['indiv2sample'][indiv_id]['samples']
            for sample_id in indiv_samples:
                tissue = self.data['sample']['sample2tissue'][sample_id]
                D[indiv_id]['sample'][tissue] = {}
                D[indiv_id]['sample'][tissue]['info'] = self.data['sample']['info'].loc[sample_id, :]
                D[indiv_id]['sample'][tissue]['expr'] = self.data['expr'].loc[sample_id, :]
                
        if save_path is not None:
            torch.save(D, save_path, pickle_protocol=4)
        return D
    
    def filter_train(self, train_indiv_file, train_tissue_file):
        tissue_whitelist = sorted(pd.read_csv(train_tissue_file, sep='\t', dtype='str', header=None).iloc[:, 0].tolist())
        indiv_whitelist = set(pd.read_csv(train_indiv_file, sep='\t', dtype='str', header=None).iloc[:, 0].tolist())
        train_indivs = []
        train_samples = []
        for sample_id in self.item2idx['sample'].keys():
            indiv_id = self.data['sample']['sample2indiv'][sample_id]
            if (indiv_id in indiv_whitelist) and (self.data['sample']['sample2tissue'][sample_id] in tissue_whitelist):
                train_indivs.append(indiv_id)
                train_samples.append(sample_id)
        train_indivs = sorted(list(set(train_indivs)))
        train_samples = sorted(list(set(train_samples)))
        return train_indivs, train_samples, tissue_whitelist
    
    def build_train_items(self, source='Any', target='Any', save_path=None):
        train_items_list = []
        tissue_pair_type = []
        for indiv_id in self.train_indivs:
            tissues = sorted(self.data['indiv']['indiv2sample'][indiv_id]['tissues'])
            for tissue_a in tissues:
                for tissue_b in tissues:
                    if tissue_a == tissue_b:
                        continue
                    if (source == 'Any' and target == 'Any') or (source == 'Any' and target == tissue_b) or (source == tissue_a and target == 'Any') or (source == tissue_a and target == tissue_b):
                        if (tissue_a not in self.train_tissues) or (tissue_b not in self.train_tissues):
                            continue
                        train_items_list.append({
                            'info': [indiv_id, tissue_a, tissue_b],
                        })
                        tissue_pair_type.append((tissue_a, tissue_b))
        train_items = {str(idx): item for idx, item in enumerate(train_items_list)}
        tissue_pair_type = list(set(tissue_pair_type))
        print(f'from {source} to {target}: {len(train_items)} items, including {len(tissue_pair_type)} directions')
        if save_path is not None:
            with open(save_path, 'w') as f:
                json.dump(train_items, f, indent=4)
        return train_items


class ReadTissuePairDataset(BuildTissuePairDataset):
    def __init__(
        self,
        train_indiv_file,
        train_tissue_file,
        expr_file,
        sample_attribute_file,
        tissue_source='Whole_Blood',
        tissue_target='Any',
        tissue_labelname='SMTSD',
        indiv_labelname='Subject_id',
    ):
        super().__init__(train_indiv_file, train_tissue_file, expr_file, sample_attribute_file, tissue_labelname, indiv_labelname)
        self.tissue_source = tissue_source
        self.tissue_target = tissue_target
        self.train_items = self.build_train_items(source=tissue_source, target=tissue_target)
        self.data['expr'] = None
    
    def __len__(self):
        return len(self.train_items)
    
    def __getitem__(self, idx):
        item = self.train_items[str(idx)]
        indiv_id, tissue_s, tissue_t = item['info']
        sample = {
            'indiv_id': self.item2idx['indiv'][indiv_id],
            'tissue_s': self.item2idx['tissue'][tissue_s],
            'tissue_t': self.item2idx['tissue'][tissue_t],
            
            'expr_s': self.data_dict[indiv_id]['sample'][tissue_s]['expr'],
            'expr_t': self.data_dict[indiv_id]['sample'][tissue_t]['expr'],
        }

        if 'info' in self.data_dict[indiv_id]['sample'][tissue_s]:
            sample['sample_info_s'] = self.data_dict[indiv_id]['sample'][tissue_s]['info']
            sample['sample_info_t'] = self.data_dict[indiv_id]['sample'][tissue_t]['info']
        
        # all expr from indiv_id
        if 'expr' in self.data_dict[indiv_id]['indiv']:
            sample['indiv_expr'] = self.data_dict[indiv_id]['indiv']['expr']
            sample['tissue_mask'] = self.data_dict[indiv_id]['indiv']['tissue_mask']
        return sample


class ZScoreByTissueOneSample:
    def __init__(self, statistics_by_tissue: dict) -> None:
        self.stats = statistics_by_tissue
        self.model = ZScore()

    def __call__(self, expr: pd.Series, tissue_idx):
        mean = self.stats[tissue_idx]["mean"].copy()
        std = self.stats[tissue_idx]["std"].copy()
        expr_transformed = self.model(
            data=np.reshape(expr.values.copy(), (1, -1)), mean=mean, std=std
        )
        return expr_transformed.astype("float32").reshape(-1)


class DataDictTransform:
    """
    transforms for expr
    """

    def __init__(
        self,
        gene_statistics,
        gene_idx,
        item2idx,
    ):
        self.gene_statistics = gene_statistics
        self.gene_idx = gene_idx
        self.item2idx = item2idx

        self.expr_transformer = ZScoreByTissueOneSample(self.gene_statistics)

    def __call__(
        self,
        data_dict,
    ):
        n_tissues = len(self.item2idx["tissue"].keys())
        n_genes = len(self.gene_idx)

        new_data_dict = {}

        for indiv_id in data_dict.keys():
            new_data_dict[indiv_id] = {
                "indiv": {
                    "info": None,
                    "tissue_mask": None,
                    "expr": None,
                },
                "sample": {},
            }

            tissues = list(data_dict[indiv_id]["sample"].keys())
            for tissue in tissues:
                tissue_idx = self.item2idx["tissue"][tissue]

                expr_input = copy.deepcopy(data_dict[indiv_id]["sample"][tissue]["expr"])
                expr_transformed = self.expr_transformer(
                    expr_input, tissue_idx
                )[self.gene_idx]

                sample_info_transformed = None

                new_data_dict[indiv_id]["sample"][tissue] = {
                    "expr": expr_transformed,
                    "info": sample_info_transformed,
                }
                if new_data_dict[indiv_id]["sample"][tissue]["info"] is None:
                    new_data_dict[indiv_id]["sample"][tissue].pop("info")

            if new_data_dict[indiv_id]["indiv"]["info"] is None:
                new_data_dict[indiv_id]["indiv"].pop("info")
            if new_data_dict[indiv_id]["indiv"]["tissue_mask"] is None:
                new_data_dict[indiv_id]["indiv"].pop("tissue_mask")
            if new_data_dict[indiv_id]["indiv"]["expr"] is None:
                new_data_dict[indiv_id]["indiv"].pop("expr")
            if new_data_dict[indiv_id]["indiv"] is None:
                new_data_dict.pop(indiv_id)
        return new_data_dict


class DataHub:
    def __init__(
        self,
        expr_file,
        dataset,
        gene_list_file,
        tissue_labelname='SMTSD',
        train_ratio=None,
        cross_validation: bool = True,
        n_folds: int = 5,
        current_fold: int = 0,
    ):
        self.dataset = dataset

        # split related
        self.cross_validation = cross_validation
        if self.cross_validation:
            assert train_ratio is None
            self.n_folds = n_folds
            self.current_fold = current_fold
        else:
            self.train_ratio = train_ratio

        self.tissue_labelname = tissue_labelname

        # read expr
        print("Loading dataset from {} ...".format(expr_file))
        self.expr = {"raw": self.read_expr(expr_file)}
        self.genes_to_use = list(
            pd.read_csv(gene_list_file, dtype="str", header=None).iloc[:, 0]
        )
        print("Number of genes to analyze: {}".format(len(self.genes_to_use)))
        genes_all = list(self.expr["raw"].columns)
        self.genes_to_use_idx = []
        for gene in list(self.genes_to_use):
            self.genes_to_use_idx.append(genes_all.index(gene))

    def setup(
        self,
    ):
        # --- split training and validation set --- #
        self.split_train_and_val()
        # ! train samples contain tissues that are not in train tissues

        # --- build statistics --- #
        self.gene_statistics, self.gene_tensor_statistics = self.build_statistics(
            input_data=self.expr["training"],
        )

        # --- set transform --- #
        self.set_transform()

        print("setup datahub over")

    def read_expr(self, file_path):
        # first column corresponds to row name
        if file_path.endswith(".ptl"):
            data = torch.load(file_path)
        elif file_path.endswith(".gz"):
            data = pd.read_csv(
                file_path,
                compression="gzip",
                sep="\t",
                converters={0: str},
                dtype="float32",
                index_col=0,
            )
        else:
            data = dt.fread(file_path, sep="\t", header=True).to_pandas()
            data = data.set_index(data.columns[0]).astype("float32")
        return data

    def split_train_and_val(self):
        individuals = self.dataset.train_indivs
        n_indiv = len(individuals)
        if self.cross_validation:
            train_indiv_indices, val_indiv_indices = self.train_val_split_cv(
                size=n_indiv, n_folds=self.n_folds, current_fold=self.current_fold
            )
        else:
            train_indiv_indices, val_indiv_indices = self.train_val_split(
                size=n_indiv, train_ratio=self.train_ratio
            )
        self.train_indivs = np.array(individuals)[train_indiv_indices]
        self.val_indivs = np.array(individuals)[val_indiv_indices]
        self.train_samples = []
        self.val_samples = []
        for indiv_id in self.train_indivs:
            samples = self.dataset.data["indiv"]["indiv2sample"][indiv_id]["samples"]
            for sample_id in samples:
                self.train_samples.append(sample_id)
        for indiv_id in self.val_indivs:
            samples = self.dataset.data["indiv"]["indiv2sample"][indiv_id]["samples"]
            for sample_id in samples:
                self.val_samples.append(sample_id)
        self.expr["training"] = self.expr["raw"].loc[self.train_samples, :]
        self.expr["validation"] = self.expr["raw"].loc[self.val_samples, :]
        tissue_feature = self.tissue_labelname
        self.tissue_labels = {
            "training": [
                self.dataset.data["sample"]["info"].loc[sample_id, tissue_feature]
                for sample_id in self.train_samples
            ],
            "validation": [
                self.dataset.data["sample"]["info"].loc[sample_id, tissue_feature]
                for sample_id in self.val_samples
            ],
        }

    def split_item_indices(self):
        # split training and validation set for dataloaders
        train_item_indices = []
        val_item_indices = []
        for key, value in self.dataset.train_items.items():
            indiv_id = value["info"][0]
            if indiv_id in self.train_indivs:
                train_item_indices.append(int(key))
            elif indiv_id in self.val_indivs:
                val_item_indices.append(int(key))
        return train_item_indices, val_item_indices

    def train_val_split(self, size: int, train_ratio: float, seed: int = 666):
        random.seed(seed)
        train_size = int(size * train_ratio)
        train_idx = random.sample(range(size), train_size)
        val_idx = list(set(range(size)) - set(train_idx))
        return train_idx, val_idx

    def train_val_split_cv(
        self, size: int, n_folds: float, current_fold: int, seed: int = 666
    ):
        random.seed(seed)
        fold_idx = random.choices(range(n_folds), k=size)
        train_idx = [i for i in range(size) if fold_idx[i] != current_fold]
        val_idx = [i for i in range(size) if fold_idx[i] == current_fold]
        return train_idx, val_idx

    def set_transform(
        self,
    ):
        self.old_data_dict = copy.deepcopy(self.dataset.data_dict)
        self.data_dict_transformer = DataDictTransform(
            gene_statistics=self.gene_statistics,
            gene_idx=self.genes_to_use_idx,
            item2idx=self.dataset.item2idx,
        )
        new_data_dict = self.data_dict_transformer(
            self.old_data_dict,
        )
        self.dataset.data_dict = new_data_dict

    def build_statistics(self, input_data: pd.DataFrame):
        tissue_feature = self.tissue_labelname
        tissue_labels = [
            self.dataset.data["sample"]["info"].loc[sample_id, tissue_feature]
            for sample_id in self.train_samples
        ]
        tissue2idx = self.dataset.item2idx["tissue"]

        def build_statistics_by_tissue(
            input_data, tissue_labels: list, tissue2idx: dict, save_path=None
        ):
            # do after split_train_and_val
            data = {}
            tissue_list = sorted(list(set(tissue_labels)))
            for tissue in tissue_list:
                tissue_idx = tissue2idx[tissue]  # int
                mask = np.array(tissue_labels) == tissue
                tissue_data = input_data.loc[mask, :]
                tissue_data = tissue_data.values
                data[tissue_idx] = {
                    "mean": np.mean(tissue_data, axis=0),
                    "std": np.std(tissue_data, axis=0),
                }
            if save_path is not None:
                torch.save(data, save_path)
            return data

        train_data = copy.deepcopy(input_data.loc[self.train_samples, :])

        gene_statistics = {}
        gene_statistics = build_statistics_by_tissue(
            input_data=train_data, tissue_labels=tissue_labels, tissue2idx=tissue2idx
        )

        gene_tensor_statistics = {}
        gene_tensor_statistics = torch.Tensor(
            np.stack(
                [
                    np.stack(
                        [
                            gene_statistics[i]["mean"],
                            gene_statistics[i]["std"],
                        ],
                        axis=0,
                    )
                    for i in range(len(gene_statistics))
                ],
                axis=0,
            )
        )
        gene_tensor_statistics = gene_tensor_statistics[
            :, :, self.genes_to_use_idx
        ]
        return gene_statistics, gene_tensor_statistics

    def build_other_statistics(self, input_data: pd.DataFrame):
        def build_statistics_by_tissue(
            input_data, tissue_labels: list, tissue2idx: dict, save_path=None
        ):
            # do after split_train_and_val
            data = {}
            tissue_list = sorted(list(set(tissue_labels)))
            for tissue in tissue_list:
                tissue_idx = tissue2idx[tissue]  # int
                mask = np.array(tissue_labels) == tissue
                tissue_data = input_data.loc[mask, :]
                tissue_data = tissue_data.values
                data[tissue_idx] = {
                    "mean": np.mean(tissue_data, axis=0),
                    "std": np.std(tissue_data, axis=0),
                }
            if save_path is not None:
                torch.save(data, save_path)
            return data

        tissue_feature = self.tissue_labelname
        train_samples_other = [
            sample_id
            for sample_id in input_data.index
            if sample_id in self.train_samples
        ]
        tissue_labels = [
            self.dataset.data["sample"]["info"].loc[sample_id, tissue_feature]
            for sample_id in train_samples_other
        ]
        tissue2idx = self.dataset.item2idx["tissue"]

        train_data = copy.deepcopy(input_data.loc[train_samples_other, :])

        gene_statistics = {}
        gene_statistics = build_statistics_by_tissue(
            input_data=train_data, tissue_labels=tissue_labels, tissue2idx=tissue2idx
        )
        return gene_statistics

    def tensorized_norm(
        self,
        expr: torch.Tensor,
        tissue_indices: torch.Tensor,
    ):
        assert expr.ndim == 2
        if self.gene_tensor_statistics.device != expr.device:
            self.gene_tensor_statistics = self.gene_tensor_statistics.to(expr.device)
        mean = self.gene_tensor_statistics[
            tissue_indices, 0, :
        ]  # [B, n_genes]
        std = self.gene_tensor_statistics[
            tissue_indices, 1, :
        ]  # [B, n_genes]

        new_std = std.clone()
        new_std[mean < 1] = 1
        normed_expr = (expr - mean) / (new_std + EPS)
        return normed_expr

    def tensorized_unnorm(
        self,
        expr: torch.Tensor,
        tissue_indices: torch.Tensor,
    ):
        """
        Convert zscore-normalized expr data to original
        coordinate system.
        """
        # expr: [B, n_genes], tissue_indices: [B]
        assert expr.ndim == 2
        if self.gene_tensor_statistics.device != expr.device:
            self.gene_tensor_statistics = self.gene_tensor_statistics.to(expr.device)
        mean = self.gene_tensor_statistics[
            tissue_indices, 0, :
        ]  # [B, n_genes]
        std = self.gene_tensor_statistics[
            tissue_indices, 1, :
        ]  # [B, n_genes]
        new_std = std.clone()
        new_std[mean < 1] = 1
        unnormed_expr = expr * (new_std + EPS) + mean
        return unnormed_expr

    def tensorized_transform(self, expr: torch.Tensor, tissue_indices: torch.Tensor):
        assert isinstance(expr, torch.Tensor)

        new_expr = self.tensorized_norm(
            expr, tissue_indices,
        )
        return new_expr

    def tensorized_untransform(self, expr, tissue_indices):
        assert isinstance(expr, torch.Tensor)
        new_expr = self.tensorized_unnorm(expr, tissue_indices)
        return new_expr
