# MTM
## Introduction

MTM (Multi-tissue Transcriptome Mapping) is a unified deep multi-task learning approach to predict tissue-specific gene expression profiles using any available tissue expression profile (such as blood gene expression) from the same donor.

## Requirements
- Python 3.8
- PyTorch 1.10.2
- Numpy 1.20.3
- Pandas 1.2.4
- scikit-learn 0.24.2

## Pre-processing
Download data from GTEx Portal:
- expression data: GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz
- sample attributes: GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt

Filter the downloaded data (expression data and sample attribute data) based on:
- tissue types: select tissues with at least 50 samples
- individuals: select individuals with at least 2 tissue samples
- genes: select genes of interests (for example, protein-coding genes)

The donor id should be added to the sample attributes file as the 'Subject_id' column. The filtered data should be saved as tab-delimited text files for model training, including:
- expr: expression data, row - sample, column - gene
- sample_attr: sample attributes, row - sample, column - attributes, including tissue type (the 'SMTSD' column) and individual id (the 'Subject_id' column)
- gene_id: filtered gene ids
- indiv_id: filtered individual ids
- tissue_type: filtered tissue types

## Train
Train the MTM model to learn the mapping between diffirent tissues with GTEx data:

Example:
```bash
python train.py \
    --input_dir ../input_dir \
    --expr GTEx_expr.txt \
    --sample_attr GTEx_sample_attributes.txt \
    --gene_id GTEx_gene_id.txt \
    --indiv_id GTEx_individual_id.txt \
    --tissue_type GTEx_tissue_type.txt \
    --device "cuda:0" \
    --output_dir ../output_dir
```
The trained model will be saved in the ../\${output_dir}/models directory.
The individuals are randomly split for training (80%) and evaluation (20%), and the individual ids are saved in the ../\${output_dir}/data_split directory.
## Predict
Utilize the trained MTM model to predict tissue-specific gene expression profiles on unseen individuals.

#### Example
Prepare the input expression profiles of a specific tissue type (source tissue), such as "Whole_Blood". We can make use of the GTEx expression data as the input expression profiles by the following steps:
- Filter the GTEx expression data based on:
    - the selected genes (GTEx_gene_id.txt)
    - individuals (../\${output_dir}/data_split/val_indivs.txt)
    - tissue type (Whole_Blood)
- Save the resulting filtered expression data as tab-delimited text file, such as GTEx_expr.val_set.Whole_Blood.txt.

Then, we can predict the expression profiles of another tissue type (target tissue), such as "Lung", with the trained MTM model (../\${output_dir}/models/model_ckpt.tar), by the following command:

```bash
python predict.py \
    --expr GTEx_expr.txt \
    --sample_attr GTEx_sample_attributes.txt \
    --gene_id GTEx_gene_id.txt \
    --indiv_id GTEx_individual_id.txt \
    --tissue_type GTEx_tissue_type.txt \
    --input_expr GTEx_expr.val_set.Whole_Blood.txt \
    --input_tissue_type "Whole_Blood" \
    --output_tissue_type "Lung" \
    --model_path ../output_dir/models/model_ckpt.tar \
    --output_expr ../output_dir/predicted/GTEx_expr.val_set.Whole_Blood.to.Lung.txt

```

