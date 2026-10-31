<h1 align="center">Machine Learning Dataset and Benchmark for Accurate T Cell Receptor-pHLA Binding Prediction</h1>

<p align="center">
  <a href="https://www.nature.com/articles/s43588-026-01073-y"><img src="https://img.shields.io/badge/Nature_Computational_Science-Resource-0B3D91?style=for-the-badge&logo=springer&logoColor=white" alt="Nature Computational Science — Resource"></a>
  &nbsp;
  <a href="https://doi.org/10.5281/zenodo.22647655"><img src="https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.22647655-1682D4?style=for-the-badge&logo=doi&logoColor=white" alt="Zenodo DOI"></a>
  &nbsp;
  <a href="https://huggingface.co/Jiadong001/Hi-TPH-PLMs"><img src="https://img.shields.io/badge/🤗_HuggingFace-Hi--TPH--PLMs-FFD21E?style=for-the-badge" alt="HuggingFace"></a>
  &nbsp;
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey?style=for-the-badge" alt="License: CC BY-NC 4.0"></a>
</p>

This repository contains the **Hi-TPH** dataset, a large-scale **Hi**erarchical dataset for **T**CR-**pH**LA binding prediction, and corresponding codes used for data collection and benchmarks, accompanying our [Resource article in Nature Computational Science](https://doi.org/10.1038/s43588-026-01073-y).

<!-- See our Research Briefing (*coming soon*) for a concise overview of the motivation, key findings, and takeaways of this work. -->



## 🌟 Overview of Hi-TPH

![overview](./assets/overview.png)


## ⚙️ Dependency

Install basic packages using `env.yaml`/`requirements.txt` or the following instructions:

```bash
conda create -n hitph python=3.8
conda activate hitph

# remove '-i https://pypi.tuna.tsinghua.edu.cn/simple/' if necessary
pip install pandas==2.0.3 numpy==1.24.3 scikit-learn tqdm jupyter notebook -i https://pypi.tuna.tsinghua.edu.cn/simple/

conda install pytorch==1.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# pip --trusted-host pypi.tuna.tsinghua.edu.cn install torch==1.13.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

pip install transformers==4.36.2 datasets==2.16.1 tokenizers==0.15.0 -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install tape_proteins biopython==1.83 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# for plotting and running notebooks
conda install matplotlib seaborn jupyter notebook
```

An alternative Python 3.10 environment (AMPLIFY needs python>=3.10):
```bash
conda create -n py310 python=3.10
conda activate py310

pip install pandas==2.0 numpy==1.26 scikit-learn tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/

# AMPLIFY needs python>=3.10, torch==2.2, xformers==0.0.24, ...
# details see: https://github.com/chandar-lab/AMPLIFY/blob/main/pyproject.toml
pip install torch==2.2.0 xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple/
pip install transformers==4.38 accelerate==0.27 deepspeed==0.13 -i https://pypi.tuna.tsinghua.edu.cn/simple/

pip install tape_proteins biopython==1.83 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# for plotting and running notebooks
conda install matplotlib seaborn jupyter notebook
```


## 🔍 Data Curation

1. [`data_preprocess`](./data_preprocess/) folder descripes detailed data collection and processing procedures.

    - check [`data_preprocess/README.md`](./data_preprocess/README.md) for instructions.

2. [`data`](./data/) details:

<table style="text-align:center">
<tr>
<th rowspan=2 style="text-align:center"></th>
<th rowspan=2 style="text-align:center">peptide</th>
<th rowspan=2 style="text-align:center">HLA<br></th>
<th colspan=6 style="text-align:center">TCR</th>
<!-- <th rowspan=2 style="text-align:center">#samples</th>
<th rowspan=2 style="text-align:center">source</th> -->
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
<!-- <td>214,641</td>
<td>[1-4]</td> -->
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
<!-- <td>78,679</td>
<td>[1-3]</td> -->
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
<!-- <td>28,262</td>
<td>[1-3]</td> -->
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
<!-- <td>26,704</td>
<td>[1-3,5]</td> -->
<tr>
<td colspan=11></td>
</tr>
<tr>
<td style="font-weight:bold"> immrep2023_solutions.csv </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<td> √ </td>
<!-- <td>3,484</td>
<td>[6]</td> -->
<tr>
<td colspan=11></td>
</tr>
<tr>
<td colspan=2 style="font-weight:bold"> Hi-TpH-hla_allele2seq.json </td>
<td colspan=9>Dictionary for mapping HLA allele to HLA amino acid sequences (customized for Hi-TPH)</td>
</tr>
<tr>
<td colspan=2 style="font-weight:bold"> Hi-TpH-tcr_gene2seq.json </td>
<td colspan=9>Dictionary for mapping TCR gene name to amino acid sequences (customized for Hi-TPH)</td>
</tr>
<tr>
<td colspan=11></td>
</tr>
<tr>
<td colspan=2 style="font-weight:bold"> hla_allele2seq.json </td>
<td colspan=9>Dictionary for mapping HLA allele to HLA amino acid sequences</td>
</tr>
<tr>
<td colspan=2 style="font-weight:bold"> tcr_gene2seq.json </td>
<td colspan=9>Dictionary for mapping TCR gene name to amino acid sequences</td>
</tr>
</table>


## 📊 Benchmarks

1. The split Level I-IV benchmark datasets are saved in [`benchmarks_dataset`](./benchmarks_dataset/) folder.

2. Run benchmarks of different levels as follows (in [`benchmarks`](./benchmarks/) folder):

    - **[Settings]** First, change the path in the following file to your own：
      - `bash` files in [`benchmarks/scripts`](./benchmarks/scripts/): ``**_path``, e.g., ``data_path``.
      - `plm_models.py` in [`benchmarks`](./benchmarks): ``**_checkpoint``, e.g., ``esm2_8m_checkpoint``.

    - **[Training]** See `benchmarks/scripts/train_**.sh` to run `train_main.py` for training models.
      - If **finetune** PLMs, you need to **add a line to the script** with the parameter ``--finetune``; if not finetune PLMs, remove it.

    - **[Evaluation]** After Training, you can test the model with `test_main.py`.
      - See `benchmarks/scripts/eval_**.sh` to run `test_main.py`

**Note: we have uploaded checkpoints of Hi-TPH-PLMs to [HuggingFace🤗: Jiadong001/Hi-TPH-PLMs](https://huggingface.co/Jiadong001/Hi-TPH-PLMs).**


## 📖 Citation

If you find Hi-TPH or Hi-TPH-PLMs useful in your research, please consider citing our paper:

```bibtex
@article{zhu2026machine,
  title={Machine learning dataset and benchmark for accurate T cell receptor--pHLA binding prediction},
  author={Zhu, Xinyuan and Lu, Jiadong and Lu, Yeqing and Zhang, Yuyan and Feng, Fuli},
  journal={Nature Computational Science},
  pages={1--11},
  year={2026},
  publisher={Nature Publishing Group US, New York}
}
```


## 👍 Acknowledgements

We thank the developers and maintainers of the public biological databases and datasets that enabled this work, including:
- [Immune Epitope Database (IEDB)](https://www.iedb.org)
- [VDJdb](https://github.com/antigenomics/vdjdb-db/releases/tag/2023-06-01)
- [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR)
- [ImmuneCODE](https://clients.adaptivebiotech.com/pub/covid-2020)
- [IMMREP2023 evaluation dataset](https://github.com/justin-barton/IMMREP23)
- IMGT (Reference amino acid sequences for [HLA alleles](https://github.com/ANHIG/IMGTHLA/blob/Latest) and [human TCR genes](https://www.imgt.org/vquest/refseqh.html))
- Open datasets from published/preprint studies (e.g., [STAPLER](https://files.aiforoncology.nl/stapler/data/))

