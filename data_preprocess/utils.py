import os
import matplotlib.pyplot as plt


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