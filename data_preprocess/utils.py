import os
import matplotlib.pyplot as plt


# Reconstruct TCR variable sequence
def reconstruct_vseq(data):
    """
    Input: "vgene_seq-cdr3-jgene_seq"
    Return: 
        - If succeed, return "reconstructed sequence"
        - If fail, return "(UNK)"
    """
    vgene_seq = data.split('-')[0]
    cdr3 = data.split('-')[1]
    jgene_seq = data.split('-')[2]

    # cut vgene seq, eg. CA: ..YL|CAVT
    cdr3_start = cdr3[:2]
    vgene_seq_end = vgene_seq.rfind(cdr3_start)                         # last position

    # cut jgene seq, eg. LQ: GKLQ|FG..
    cdr3_end = cdr3[-2:]
    jgene_seq_start = jgene_seq.find(cdr3_end) + len(cdr3_end)          # first position + 2    

    if (vgene_seq_end != -1) & (jgene_seq_start != (-1+len(cdr3_end))): # + len(cdr3_end) !!!
        return vgene_seq[:vgene_seq_end] + cdr3 + jgene_seq[jgene_seq_start:]
    else:
        return '(UNK)'


def save_csv(df, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    df.to_csv(path+filename, index=False)
    print(f"File {path+filename} has been successfully saved.")


def plot_non_empty_percentage(df, database):
    non_empty_counts = df.count()
    total_rows = len(df)
    non_empty_percentages = (non_empty_counts / total_rows) * 100

    plt.figure(figsize=(10, 6))
    non_empty_percentages.plot(kind='bar')
    plt.title(f'{database}: Percentage of Non-Empty Elements in Each Column')
    plt.xlabel('Columns')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=75)
    plt.hlines(100, 0-0.5, len(df.columns)+0.5, color='black', linestyles='dashed')
    plt.tight_layout()
    plt.show()