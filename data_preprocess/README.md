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


# Description of Raw Data

- [] To be updated..

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
- Peptides are included in the first colums by default.

Notes:
- DeepTCR 
    - Data sources are very diverse. (VDJdb, McPas-TCR...)
    - Mainly use TCR data for TCR featurization.
    - Follow-up work: 
        - [DeepTCR_Cancer](https://github.com/sidhomj/DeepTCR_Cancer): TCR-peptide-HLA
        - [DeepTCR_COVID19](https://github.com/sidhomj/DeepTCR_COVID19): TCR-peptide
- TCRAI
    - `CNN-prediction-with-REGN-pilot-version-2.csv`: processed by ICON, and raw data was from 10x Genomics initial report. Model training. 9 pmhc pairs
    - `public_TCRs.csv`: from VDJdb, MaPas-TCR. 13 pmhc pairs
    - "pMHC-specific/peptide-specific": one peptide <-> one HLA allele -> TCR samples
        - similar to "allele-specific"
    - HLA: 1-2 fields
    - *ICON workflow*: to robustly identify reliable binding events from these high-throughput TCR-pMHC binding data.
        - *May be useful for us when processing our dataset.*
- NetTCR-2.0
    - The dataset is a little confusing. 
    - "peptide-specific"
- NetTCR-2.1
    - Adding CDR1 and CDR2.
    - The full-length TCR was reconstructed from the V/J genes and CDR3 sequence
        - V/J genes -> CDR1 and CDR2 ?
    - V/J genes and HLA informaton are not used.
- ERGO-II
    - ERGO-I uses only the TCRβ CDR3 sequence and the peptide sequence for the binding prediciton.
    - information of TRA can be partially missed.
    - Data is not available. (McPAS-TCR, VDJdb)
    - MHC-I/II correspond to two models.
- pMTnet
    - HLA: 1-3 fields
    - `training_data.csv`, `testing_data.csv`: all positive samples
    - randomly select negative TCRs from `pMTnet/library/bg_tcr_library/TCR_output_10k.csv` to genrate negative samples. (10 times more than positive samples)
- epiTCR
    - HLA: 2-4 fields
    - proportion of positive samples: ~3%
- TAPIR
    - *novel target validation dataset*: We then further filtered the novel target validation dataset to remove any TCRs with “Confidence Score” = 0.


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
|           |            |             |            |             |             |             |            |            |            |            |
|w experiment data|$\checkmark$|$\checkmark$ |            |             |             |             |$\checkmark$|            |            |            |
|w reference|$\checkmark$|$\checkmark$ |            |             |             |             |            |            |            |            |
- "w reference" indicates that data gained from experiment has a reference.

Notes:
- DLpTCR: 
    - IEDB, McPAS-TCR, Covid-19 from VDJdb: independent set
    - pad all peptide-CDR3 pair sequences to the maximum length of 29.
- DeepTCR: 
    - TCR length<40
    - validation: McPAS-TCR
- TCRAI
    - REGN: ICON-processed 10x; Public: VDJdb & McPAS-TCR
    - For a set of CDR3s, we define the CDR3 motif length L as the longest CDR3 in the set.
- netTCR
    - MIRA: Evaluation
    - The CDR sequences were zeropadded to a maximum length of 30; HLA-A*02:01 restricted peptides were of length 9.
- PanPep
    - MIRA: Evaluation
    - Notably, **TITAN17** was excluded because this model removed peptides with few known binding TCRs and is trained on specific **COVID-19 data**
    - The published large cohort COVID19 dataset is available at https://clients.adaptivebiotech.com/pub/covid-2020. 
    - The collected 3D crystal complexes are available at PDB (https://www.rcsb.org) and their accession numbers were provided in Supplementary Data 8.
- TEIM
    - ImmuneCODE (https:// clients.adaptivebiotech.com/pub/covid-2020) == MIRA
    - The structures of TCRepitope complexes were downloaded from STCRDab (https://opig.stats.ox.ac.uk/webapps/stcrdab/Browser?all=true#downloads)

High-throughput determination of the antigen specificities of T cell receptors in single cells.
- used by DLpTCR and pMTnet


**Common data sources:**
- [x] [IEDB](https://www.iedb.org/)
- [x] [VDJdb](https://github.com/antigenomics/vdjdb-db/releases)
- [x] [McPAS-TCR](http://friedmanlab.weizmann.ac.il/McPAS-TCR/)
- [x] [ImmuneCODE-MIRA](https://clients.adaptivebiotech.com/pub/covid-2020)
- [ ] [10x Genomics](https://www.10xgenomics.com/resources/datasets)
- [ ] [PIRD](https://db.cngb.org/pird/)
- [ ] TBAdb

