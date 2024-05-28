# Data description

<table style="text-align:center">
<tr>
<th rowspan=2 style="text-align:center">Dataset</th>
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
<td>720,038</td>
<td>[1-4]</td>
</tr>
<tr>
<td style="font-weight:bold"> Hi-TpH-level-IIA.csv </td>
<td> √ </td>
<td> √ </td>
<td></td>
<td> √ </td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>78,679</td>
<td>[1-3]</td>
</tr>
<tr>
<td style="font-weight:bold"> Hi-TpH-level-IIB.csv </td>
<td> √ </td>
<td></td>
<td> √ </td>
<td> √ </td>
<td></td>
<td></td>
<td></td>
<td></td>
<td>28,375</td>
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
<td>28,262</td>
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
<td>26,704</td>
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