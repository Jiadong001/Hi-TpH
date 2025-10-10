# Machine Learning Dataset and Benchmark for Accurate T Cell Receptor-pHLA Binding Prediction

This repository contains the **Hi-TpH** dataset, a large-scale **Hi**erarchical dataset for **T**CR-**pH**LA binding prediction, and corresponding codes used for data collection and benchmarks.



## Overview of Hi-TpH

![overview](./assets/overview.png)


## Dependency

Install basic packages using `env.yaml`/`requirements.txt` or the following instructions:

```bash
conda create -n hitph python=3.8
conda activate hitph

pip install pandas==2.0.3 numpy==1.24.3 scikit-learn tqdm jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple/

conda install pytorch==1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# pip --trusted-host pypi.tuna.tsinghua.edu.cn install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install transformers==4.36.2 datasets==2.16.1 tokenizers==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install tape_proteins biopython==1.83

# for plotting
conda install matplotlib seaborn
```


## Data Curation

> [`./data_preprocess`](./data_preprocess/) -> [`./data`](./data/) 

[`./data_preprocess`](./data_preprocess/) folder descripes detailed data collection and processing procedures.

[`./data`](./data/) details:

<table style="text-align:center">
<tr>
<th rowspan=2 style="text-align:center"></th>
<th rowspan=2 style="text-align:center">peptide</th>
<th rowspan=2 style="text-align:center">HLA<br></th>
<th colspan=6 style="text-align:center">TCR</th>
<th rowspan=2 style="text-align:center">#samples</th>
<th rowspan=2 style="text-align:center">source</th>
</tr>
<tr>
<th>CDR3a</th>
<th>CDR3b</th>
<th>TRAV/J</th>
<th>TRBV/J</th>
<th>Va.seq</th>
<th>Vb.seq</th>
</tr>
<tr>
<td style="font-weight:bold"> Hi-TpH-level-I.csv </td>
<td> √ </td>
<td></td>
<td></td>
<td> √ </td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>204,376</td>
<td>[1-4]</td>
</tr>
<tr>
<td style="font-weight:bold"> Hi-TpH-level-II.csv </td>
<td> √ </td>
<td> √ </td>
<td></td>
<td> √ </td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>73,553</td>
<td>[1-3]</td>
</tr>
<tr>
<td style="font-weight:bold"> Hi-TpH-level-III.csv </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>26,353</td>
<td>[1-3]</td>
</tr>
<tr>
<td style="font-weight:bold"> Hi-TpH-level-IV.csv </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td>24,639</td>
<td>[1-3,5]</td>
<tr>
<td colspan=11></td>
</tr>
<tr>
<td colspan=2 style="font-weight:bold"> Hi-TpH-hla_allele2seq.json </td>
<td colspan=9>A dictionary for mapping from HLA allele to HLA amino acid sequences</td>
</tr>
<tr>
<td colspan=2 style="font-weight:bold"> Hi-TpH-tcr_gene2seq.json </td>
<td colspan=9>A dictionary for mapping from TCR gene name to amino acid sequences</td>
</tr>
</table>

- [1] [IEDB](https://www.iedb.org/)
- [2] [VDJdb](https://github.com/antigenomics/vdjdb-db/releases)
- [3] [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/)
- [4] [ImmuneCODE-MIRA](https://clients.adaptivebiotech.com/pub/covid-2020)
- [5] [STAPLER data](https://files.aiforoncology.nl/stapler/data/)


## Run Benchmarks

We save the splited level1~level4 benchmark datasets in [`./benchmarks_dataset`](./benchmarks_dataset/) .

- You can also prepare benchmark datasets from scratch (see `prepare_benchmark_data.ipynb`).


Train models using benchmark datasets of different levels (see [`./benchmarks`](./benchmarks/)):

- [Settings] First, change the path in the following file to your own：
  - `bash` files in [`./benchmarks/scripts`](./benchmarks/scripts/) folder: ``**_path``, e.g., ``data_path``.
  - `data_loader.py` and `plm_models.py`: ``**_checkpoint``, e.g., ``esm2_8b_checkpoint``.

- [Training] See `./benchmarks/scripts/train_**.sh` to run `train_main.py` for training models.
  - If **finetune** PLMs, you need to **add a line to the script** with the parameter ``--finetune``; if not finetune PLMs, remove it.

- [Evaluation] After Training, you can test the model with `test_main.py`.
  - See `./benchmarks/scripts/eval_**.sh` to run `test_main.py`
  - Hint: the evaluation in `train_main.py` is only for the last epoch of models, not the best epoch.


> Define the max_len for different levels.：
>|           | pep_max_len | tcr_max_len |
>| :-------: | :---------: | :---------: |
>|  level-I  |     15      |     19      |
>| level-II | 44 (10+34)  |     19      |
>| level-III |     10      |     19      |
>| level-IV  |     10      |     121     |
