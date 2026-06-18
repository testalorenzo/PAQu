#
# Single combined figure for the real-data DE simulation (10 vs 10, seed=0),
# one row per effect size D_size in {0.33, 0.66, 1}:
#   (a) ROC curves for DE-isoform detection -- PAQu, IsoBayes + DESeq2 /
#       edgeR / limma (on the exp-transformed Ihat, see
#       PAQu_simulation_compare_de_real.py).
#   (b) Hexbin of log2(IsoBayes Ihat) vs PAQu's Ihat -- per-(block, iso,
#       sample) isoform abundance estimates from sim_results_isobayes_real_exp_*.csv
#       and app_results_PAQu_real_*.csv.
#

import ast
import sys
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _utils import roc_curve, roc_auc_score, set_size
from compare_methods import load_de_long_real

SEEDS = [0]
D_SIZES = [0.33, 0.66, 1]
GROUP_SIZE = 5
GROUP_TAG = '' if GROUP_SIZE == 10 else f'_{GROUP_SIZE}v{GROUP_SIZE}'

ROC_METHODS = [
    'PAQu',
    'IsoBayes + DESeq2',
    'IsoBayes + edgeR',
    'IsoBayes + limma',
]


def explode_ihat(df, value_name):
    df = df.copy()
    df['iso'] = df['iso'].apply(ast.literal_eval)
    df['Ihat'] = df['Ihat'].apply(ast.literal_eval)
    rows = []
    for _, r in df.iterrows():
        Ihat = np.array(r['Ihat'])  # n x q
        n, q = Ihat.shape
        for j, iso in enumerate(r['iso']):
            for k in range(n):
                rows.append((r['block'], iso, k, Ihat[k, j]))
    return pd.DataFrame(rows, columns=['block', 'iso', 'sample', value_name])


def load_abundance_long(seed, D_size):
    suffix = f'seed{seed}_D{D_size}{GROUP_TAG}'

    paqu = pd.read_csv(f'app_results_PAQu_real_{suffix}.csv')
    isobayes = pd.read_csv(f'sim_results_isobayes_real_exp_{suffix}.csv')

    paqu_long = explode_ihat(paqu, 'Ihat_paqu')
    iso_long = explode_ihat(isobayes, 'Ihat_isobayes')

    merged = paqu_long.merge(iso_long, on=['block', 'iso', 'sample'], how='inner')
    merged['seed'] = seed
    merged['D_size'] = D_size

    return merged


if __name__ == '__main__':

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook")
    width = 500.484

    de_df = pd.concat(
        [load_de_long_real(seed, D_size) for seed, D_size in itertools.product(SEEDS, D_SIZES)],
        ignore_index=True
    )
    de_df = de_df[de_df['method'].isin(ROC_METHODS)]

    abund_df = pd.concat(
        [load_abundance_long(seed, D_size) for seed, D_size in itertools.product(SEEDS, D_SIZES)],
        ignore_index=True
    )
    abund_df['log2_Ihat_isobayes'] = np.log2(abund_df['Ihat_isobayes'])

    single_w, single_h = set_size(width, subplots=(1, 2))
    fig, axes = plt.subplots(len(D_SIZES), 2, figsize=(single_w * 2.2, single_h * 1.8 * len(D_SIZES)))

    for i, d in enumerate(D_SIZES):

        # (a) ROC curves
        ax = axes[i, 0]
        sub = de_df[de_df['D_size'] == d]
        auc_lines = []
        for method in ROC_METHODS:
            group = sub[sub['method'] == method]
            fpr, tpr = roc_curve(group['is_DE'], -group['score'])
            auc = roc_auc_score(group['is_DE'], -group['score'])
            ax.plot(fpr, tpr, label=method)
            auc_lines.append(f'{method}: AUC={auc:.2f}')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        ax.text(0.98, 0.02, '\n'.join(auc_lines), transform=ax.transAxes,
                ha='right', va='bottom', fontsize='x-small',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_xlabel('False positive rate')
        ax.set_ylabel('True positive rate')
        ax.set_title(f'(a) DE-isoform detection (D = {d})')

        # (b) Hexbin: log2(IsoBayes Ihat) vs PAQu Ihat
        ax = axes[i, 1]
        sub_abund = abund_df[abund_df['D_size'] == d]
        hb = ax.hexbin(sub_abund['log2_Ihat_isobayes'], sub_abund['Ihat_paqu'], gridsize=40, cmap='viridis',
                        mincnt=1, bins='log')
        fig.colorbar(hb, ax=ax, label='log10(Count)')
        ax.set_xlim(left=-2.5)
        ax.set_ylim(bottom=-5)
        ax.set_xlabel('log2(IsoBayes est. isoform abundance)')
        ax.set_ylabel('PAQu est. isoform abundance')
        ax.set_title(f'(b) Isoform abundance estimates (D = {d})')

        corr = np.corrcoef(sub_abund['log2_Ihat_isobayes'], sub_abund['Ihat_paqu'])[0, 1]
        print(f'D_size = {d}: Pearson correlation (log2 IsoBayes Ihat vs PAQu Ihat) = {corr:.3f}')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize='small')
    fig.suptitle(f'Real-data simulation ({GROUP_SIZE} vs {GROUP_SIZE}): DE detection and abundance estimates', y=1.01)
    plt.tight_layout()
    out_path = f'plots/DE_combined_real{GROUP_TAG}.pdf'
    plt.savefig(out_path, bbox_inches="tight")

    print(f'\nSaved {out_path}')
