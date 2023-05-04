from munch import Munch
import argparse
import torch

from data import DataHub
from tmp_dataset import ReadTissuePairDataset
from model import MTM

if __name__ == "__main__":
    device = torch.device("cuda:0")

    parser = argparse.ArgumentParser('argument for prediction')
    # * training-related data
    parser.add_argument('--input_dir', type=str, help='the input directory')
    parser.add_argument('--expr', type=str, help='the expression file for model training in the input dir')
    parser.add_argument('--sample_attr', type=str, help='the sample attribute file for model training in the input dir')
    parser.add_argument('--gene_id', type=str, help='the gene id file for model training in the input dir')
    parser.add_argument('--indiv_id', type=str, help='the individual id file for model training in the input dir')
    parser.add_argument('--tissue_type', type=str, help='the tissue type file for model training in the input dir')
    parser.add_argument('--device', type=str, default="cuda:0", help='device to use')

    # * prediction-related data
    parser.add_argument('--model_path', type=str, help='the prediction model path')
    parser.add_argument('--input_expr', type=str, help='the input expression file for prediction')
    parser.add_argument('--input_tissue_type', type=str, help='the tissue source, which is the tissue type of the input expression file')
    parser.add_argument('--output_tissue_type', type=str, help='the tissue target, which is the tissue type of the output expression file')
    parser.add_argument('--output_expr', type=str, help='the output expression file path')

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

    model = MTM(
        n_genes=len(dh.genes_to_use),
        n_tissues=len(dh.dataset.item2idx["tissue"]),
        device=device,
    )
    model = model.to(device)
    model.setup_optimizers()
    model.load_ckpt(opt.model_path)

    # * Predict the expression profiles on unseen individuals
    expr_tpm = pd.read_csv(os.path.join(opt.input_dir, opt.input_expr), sep="\t", index_col=0, header=True)
    expr_tpm = expr_tpm.loc[:, dh.genes_to_use]
    expr_pred =  model.predict(
        expr_tpm,
        tissue_source=opt.input_tissue_type,
        tissue_target=opt.output_tissue_type,
        datahub=dh,
    )
    expr_pred.to_csv(opt.output_expr, sep="\t", index=True, header=True)
