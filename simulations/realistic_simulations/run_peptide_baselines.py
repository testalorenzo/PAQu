#
# Fig 2C-style "naive" peptide-abundance baselines for the real-data DE
# simulation: per subset isoform, a two-sample t-test (group A == 0 vs
# A == 1) on the average / sum / maximum of the peptides mapped to that
# isoform (subset_isoforms.csv's mapped_peptides).
#
# Adapted from ../PAQu_simulations_peptide_baselines.py, but reads the
# already-perturbed real-data subset (no data regeneration needed) -- runs
# in seconds.
#
# Output (per replicate): sim_results_peptide_baselines_real_seed{S}_D{D}.csv
#   columns: block, iso, pval_avg, pval_sum, pval_max
#

import ast
import csv
import itertools

import numpy as np
import pandas as pd
import scipy.stats as stats

SEEDS = [0]
D_SIZES = [0.33, 0.66, 1]
GROUP_SIZE = 5
GROUP_TAG = '' if GROUP_SIZE == 10 else f'_{GROUP_SIZE}v{GROUP_SIZE}'

COLUMNS = ['block', 'iso', 'pval_avg', 'pval_sum', 'pval_max']


if __name__ == '__main__':

    subset_isoforms = pd.read_csv('real_sim_data/subset_isoforms.csv')
    subset_isoforms['mapped_peptides'] = subset_isoforms['mapped_peptides'].apply(ast.literal_eval)

    for seed, D_size in itertools.product(SEEDS, D_SIZES):
        suffix = f'seed{seed}_D{D_size}{GROUP_TAG}'

        sample_assignment = pd.read_csv(f'real_sim_data/sample_assignment_{suffix}.csv', index_col=0)
        order = sample_assignment.index.tolist()
        A = sample_assignment['group'].values

        P = pd.read_csv(f'real_sim_data/P_perturbed_{suffix}.csv', index_col=0).loc[order]

        out_path = f'sim_results_peptide_baselines_real_{suffix}.csv'
        with open(out_path, 'w') as fd:
            writer = csv.writer(fd)
            writer.writerow(COLUMNS)

        rows = []
        for _, r in subset_isoforms.iterrows():
            Pi = P.loc[:, r['mapped_peptides']].values
            avg = Pi.mean(axis=1)
            summ = Pi.sum(axis=1)
            mx = Pi.max(axis=1)
            rows.append({
                'block': r['block'],
                'iso': r['iso'],
                'pval_avg': stats.ttest_ind(avg[A == 0], avg[A == 1]).pvalue,
                'pval_sum': stats.ttest_ind(summ[A == 0], summ[A == 1]).pvalue,
                'pval_max': stats.ttest_ind(mx[A == 0], mx[A == 1]).pvalue,
            })

        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, index=False)
        print(f'Done: {suffix} -> {out_path}')
