# Order of data preprocessing

Collect from common databases
- `collect***.ipynb`

See data provided by other papers
- `see***.ipynb`

Merge collected data from three common databases(IEDB, VDJdb, McPAS-TCR)
- `merge_main_dataset.ipynb`

Get allele/gene2sequence dictionary
- `hla_tcr_seq_dict.ipynb`

Clean Merged data
- `clean_main_dataset.ipynb`

Split collected into different levels
- `split_level.ipynb`


# Details of Raw Data

- [] To be updated...

## ¶ Data from papers

|*Level of Dataset*|DLpTCR|DeepTCR|TCRAI|ERGO-II|netTCR2.0|netTCR2.1|pMTnet|PanPep|TEIM|STAPLER|epiTCR|TAPIR|TEPCAM|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|CDR3b|✅|     |     ||✅|     |      |✅|✅||✅||✅|
|CDR3a|✅|     |     ||✅|
|CDR3ab|✅|    |     ||✅| 
|CDR3ab VJgene||✅|
|CDR3b+HLA||     |     |     |     |     |✅||||✅|
|CDR3ab VJgene+HLA||     |✅|✅||||||||✅|
|CDR123ab VJgene+HLA||     ||| |✅|
|Vab + HLA||||||||||✅|
|||
|License|||License for Non-Commercial Use of TCRAI code|MIT license|||GPL-2.0 license|GPL-3.0 license|MIT license|Apache-2.0 license|MIT License|
- Peptides are included in the first colums by default.


## ¶ Data from database

| *Database*|   DLpTCR   |   DeepTCR   |   TCRAI    |   ERGO-II   |   netTCR2.0 |   netTCR2.1 |   pMTnet   |   PanPep   |   TEIM     |   epiTCR   |TAPIR|TEPCAM|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|IEDB       |$\checkmark$|             |            |             |$\checkmark$ |$\checkmark$ |            |$\checkmark$|            |$\checkmark$||$\checkmark$|
|VDJdb      |$\checkmark$|             |$\checkmark$|$\checkmark$ |$\checkmark$ |$\checkmark$ |$\checkmark$|$\checkmark$|$\checkmark$|$\checkmark$|$\checkmark$ |$\checkmark$|
|McPAS-TCR  |$\checkmark$|             |$\checkmark$|$\checkmark$ |             |$\checkmark$ |$\checkmark$|$\checkmark$|$\checkmark$|$\checkmark$||$\checkmark$|
|ImmuneCODE-MIRA|        |             |            |             |$\checkmark$ |             |$\checkmark$|            |$\checkmark$|            |
|10xGenomics|            |$\checkmark$ |$\checkmark$|             |$\checkmark$ |$\checkmark$ |            |            |            |$\checkmark$|
|PIRD       |            |             |            |             |             |             |$\checkmark$|$\checkmark$|            |            |
|TBAdb      |            |             |            |             |             |             |            |            |            |$\checkmark$|


**Accessed common data sources:**
- [x] [IEDB](https://www.iedb.org/)
- [x] [VDJdb](https://github.com/antigenomics/vdjdb-db/releases)
- [x] [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/)
- [x] [ImmuneCODE-MIRA](https://clients.adaptivebiotech.com/pub/covid-2020)
- [ ] [10x Genomics](https://www.10xgenomics.com/resources/datasets)
- [ ] [PIRD](https://db.cngb.org/pird/), TBAdb

