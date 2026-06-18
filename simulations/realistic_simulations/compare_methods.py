#
# Compare PAQu, IsoBayes and the Fig 2C-style peptide-abundance t-test
# baselines on DE-isoform detection, for the semi-synthetic real-data
# simulation (10 vs 10 real Control samples, ~10k-peptide subset, ~15% of
# subset isoforms truly DE per replicate).
#
# Ground truth: real_sim_data/ground_truth_seed{S}_D{D}.csv (is_DE per
# (block, iso)). Scores (all "smaller = more evidence of DE"):
#   - PAQu's LFSR_D                                  (app_results_PAQu_real_*.csv)
#   - IsoBayes's per-isoform t-test p-value          (sim_results_isobayes_real_exp_*.csv)
#   - IsoBayes's Ihat + edgeR/DESeq2/limma/ALDEx2     (sim_results_isobayes_real_exp_*.csv)
#   - t-test on avg/sum/max of mapped peptides       (sim_results_peptide_baselines_real_*.csv)
#
# IsoBayes results are read from the "_exp" CSVs (PAQu_simulations_isobayes_real.py
# --exp): P_values.csv/T_values.csv are log2-scale, but IsoBayes's "psm" Y
# input and tpm prior expect linear-scale counts/TPM, so the inputs are
# exponentiated (2**x) before fitting IsoBayes. PAQu's own input is
# unaffected (it is fit on the original log-scale data).
#
# Isoforms are pooled across the 3 seeds within each D_size for ROC power.
#

import ast
import sys
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from _utils import roc_curve, roc_auc_score, set_size

SEEDS = [0]
D_SIZES = [0.33, 0.66, 1]
GROUP_SIZE = 5
GROUP_TAG = '' if GROUP_SIZE == 10 else f'_{GROUP_SIZE}v{GROUP_SIZE}'


def load_de_long_real(seed, D_size):
    suffix = f'seed{seed}_D{D_size}{GROUP_TAG}'

    gt = pd.read_csv(f'real_sim_data/ground_truth_{suffix}.csv')

    paqu = pd.read_csv(f'app_results_PAQu_real_{suffix}.csv')
    paqu['iso'] = paqu['iso'].apply(ast.literal_eval)
    paqu['LFSR_D'] = paqu['LFSR_D'].apply(ast.literal_eval)
    paqu_exp = paqu[['block', 'iso', 'LFSR_D']].explode(['iso', 'LFSR_D'], ignore_index=True)
    paqu_exp['score'] = paqu_exp['LFSR_D'].astype(float)
    paqu_exp['method'] = 'PAQu'
    paqu_exp = paqu_exp[['block', 'iso', 'score', 'method']]

    isobayes = pd.read_csv(f'sim_results_isobayes_real_exp_{suffix}.csv')
    isobayes['iso'] = isobayes['iso'].apply(ast.literal_eval)
    isobayes['pval_isobayes'] = isobayes['pval_isobayes'].astype(str).str.replace('nan', "float('nan')").apply(eval)
    iso_exp = isobayes[['block', 'iso', 'pval_isobayes']].explode(['iso', 'pval_isobayes'], ignore_index=True)
    iso_exp['score'] = iso_exp['pval_isobayes'].astype(float)
    iso_exp['method'] = 'IsoBayes (t-test)'
    iso_exp = iso_exp[['block', 'iso', 'score', 'method']]

    isobayes_de_method_names = {
        'pval_edger': 'IsoBayes + edgeR',
        'pval_deseq2': 'IsoBayes + DESeq2',
        'pval_deseq2_sf1': 'IsoBayes + DESeq2 (sf=1)',
        'pval_limma': 'IsoBayes + limma',
        'pval_aldex2': 'IsoBayes + ALDEx2',
    }
    isobayes_de_exps = []
    for col, method in isobayes_de_method_names.items():
        isobayes[col] = isobayes[col].astype(str).str.replace('nan', "float('nan')").apply(eval)
        exp = isobayes[['block', 'iso', col]].explode(['iso', col], ignore_index=True)
        exp['score'] = exp[col].astype(float)
        exp['method'] = method
        isobayes_de_exps.append(exp[['block', 'iso', 'score', 'method']])

    peptide = pd.read_csv(f'sim_results_peptide_baselines_real_{suffix}.csv')
    peptide_method_names = {
        'pval_avg': 't-test (avg peptide)',
        'pval_sum': 't-test (sum peptide)',
        'pval_max': 't-test (max peptide)',
    }
    pep_exps = []
    for col, method in peptide_method_names.items():
        exp = peptide[['block', 'iso', col]].copy()
        exp['score'] = exp[col].astype(float)
        exp['method'] = method
        pep_exps.append(exp[['block', 'iso', 'score', 'method']])

    de_long = pd.concat([paqu_exp, iso_exp] + isobayes_de_exps + pep_exps, ignore_index=True)
    de_long['score'] = de_long['score'].fillna(1.0)

    de_long = de_long.merge(gt[['block', 'iso', 'is_DE']], on=['block', 'iso'], how='left')
    de_long['is_DE'] = de_long['is_DE'].astype(bool)
    de_long['seed'] = seed
    de_long['D_size'] = D_size

    return de_long


if __name__ == '__main__':

    sns.set_theme(style="whitegrid")
    sns.set_context("notebook")
    width = 500.484

    df = pd.concat(
        [load_de_long_real(seed, D_size) for seed, D_size in itertools.product(SEEDS, D_SIZES)],
        ignore_index=True
    )

    print(f'Loaded {len(df)} (block, iso, method) rows '
          f'({df.groupby(["seed", "D_size"])["block"].count().iloc[0]} rows per (seed, D_size))')

    #
    # ROC curves: score is a p-value-like quantity, so a sample is called DE
    # when score < threshold -- i.e. (1 - score) acts as the "DE probability".
    # Faceted by true effect size D_size, pooling the 3 seeds for power.
    #

    d_sizes = sorted(df['D_size'].unique())

    single_w, single_h = set_size(width, subplots=(1, 3))
    fig, axes = plt.subplots(1, len(d_sizes), figsize=(single_w * len(d_sizes), single_h), sharex=True, sharey=True)
    if len(d_sizes) == 1:
        axes = [axes]

    auc_table = []
    for j, d in enumerate(d_sizes):
        ax = axes[j]
        sub = df[df['D_size'] == d]
        for method, group in sub.groupby('method'):
            fpr, tpr = roc_curve(group['is_DE'], -group['score'])
            auc = roc_auc_score(group['is_DE'], -group['score'])
            ax.plot(fpr, tpr, label=method)
            auc_table.append((d, method, auc))
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        ax.set_title(f'D = {d:.2f}')
        ax.set_xlabel('False positive rate')
        if j == 0:
            ax.set_ylabel('True positive rate')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize='small')
    fig.suptitle(f'DE-isoform detection: ROC curves (real-data simulation, {GROUP_SIZE} vs {GROUP_SIZE})')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(f'plots/DE_roc_real{GROUP_TAG}.pdf', bbox_inches="tight")

    auc_table = pd.DataFrame(auc_table, columns=['D_size', 'method', 'AUC'])
    print('\nAUC by method and D_size (pooled across 3 seeds):')
    for d in d_sizes:
        print(f'D_size = {d:.2f}')
        cell = auc_table[auc_table['D_size'] == d]
        for _, row in cell.iterrows():
            print(f'  {row["method"]:<22s} AUC = {row["AUC"]:.3f}')

    #
    # TPR / FPR at the conventional alpha = 0.05 threshold, by method and D_size
    #

    alpha = 0.05
    df['called_DE'] = df['score'] < alpha

    rates = df.groupby(['method', 'D_size', 'is_DE'])['called_DE'].mean().reset_index()
    rates['rate_type'] = rates['is_DE'].map({True: 'TPR (sensitivity)', False: 'FPR'})

    g = sns.catplot(
        data=rates, kind='bar', x='D_size', y='called_DE', hue='method',
        col='rate_type', palette='pastel',
        height=set_size(width, subplots=(1, 1))[1], aspect=1.1
    )
    g.set_axis_labels('True effect size', f'Rate (alpha = {alpha})')
    g.set_titles("{col_name}")
    for ax in g.axes.flat:
        ax.set_ylim(0, 1)
        ax.axhline(alpha, linestyle='--', color='grey', linewidth=1)
    g.figure.suptitle(f'DE-isoform detection: TPR / FPR at alpha = 0.05 (real-data simulation, {GROUP_SIZE} vs {GROUP_SIZE})', y=1.05)
    plt.savefig(f'plots/DE_tpr_fpr_real{GROUP_TAG}.pdf', bbox_inches="tight")

    print(f'\nTPR / FPR at alpha = {alpha} (by method, D_size):')
    print(df.groupby(['method', 'D_size', 'is_DE'])['called_DE'].mean().rename('rate'))
