#
# PAQu fitting on the 20-sample (10 vs 10), ~10k-peptide perturbed subset
# of the real dataset (one run per (seed, D_size) replicate).
#
# Adapted from ../PAQu_app.py's per-block fx, using paqu_blocks helpers,
# restricted to the blocks in real_sim_data/subset_blocks.txt.
#
# Usage:
#   python PAQu_app_real.py                 # run all 9 (seed, D_size) replicates
#   python PAQu_app_real.py <seed> <D_size> # run a single replicate (smoke test)
#
# Output (per replicate): app_results_PAQu_real_seed{S}_D{D}.csv
#   columns: block, iso, n, q, r, time, Dhat, LFSR_D, Ihat, I0hat
#

import sys
import csv
import itertools

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from _blocks import load_annotations_and_blocks, get_block_mask
from PAQu import PAQu

SEEDS = [0]
D_SIZES = [0.33, 0.66, 1]
GIBBS_ITERS = 3000
BURN_IN = 2000
N_JOBS = 31
GROUP_SIZE = 5
GROUP_TAG = '' if GROUP_SIZE == 10 else f'_{GROUP_SIZE}v{GROUP_SIZE}'

COLUMNS = ['block', 'iso', 'n', 'q', 'r', 'time', 'Dhat', 'LFSR_D', 'Ihat', 'I0hat']


def run_replicate(seed, D_size, Z, assign_block_I, collapsed_dict, T, subset_blocks):

    suffix = f'seed{seed}_D{D_size}{GROUP_TAG}'

    sample_assignment = pd.read_csv(f'real_sim_data/sample_assignment_{suffix}.csv', index_col=0)
    order = sample_assignment.index.tolist()
    A = sample_assignment['group'].values

    P = pd.read_csv(f'real_sim_data/P_perturbed_{suffix}.csv', index_col=0).loc[order]
    T_rep = T.loc[order]

    out_path = f'app_results_PAQu_real_{suffix}.csv'
    with open(out_path, 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(COLUMNS)

    def fx(block):

        np.random.seed(block + seed * 1000)

        mask, iso_ordered, dropped = get_block_mask(Z, assign_block_I, block, collapsed_dict)

        T_block_local = T_rep.loc[:, iso_ordered + dropped].copy()
        for dropped_iso in dropped:
            representative_iso = list(collapsed_dict[dropped_iso])
            preserved = [i for i in representative_iso if i in iso_ordered]
            T_block_local.loc[:, preserved] = T_block_local[representative_iso].mean(axis=1).values.reshape(-1, 1)

        T_block = T_block_local.loc[:, iso_ordered].values
        P_block = P.loc[:, mask.columns].values
        mask_values = mask.values

        n = T_block.shape[0]
        q = T_block.shape[1]
        r = P_block.shape[1]

        # Regress T on intercept + A, take orthogonal residuals (remove A effect only)
        projection = np.linalg.lstsq(np.vstack([np.ones(n), A]).T, T_block, rcond=None)[0]
        T_block = T_block - np.outer(A, projection[1, :])
        T_block = T_block.reshape(n, q)

        paqu = PAQu(A, mask_values, P_block, T_block, None, None)
        paqu.update_hyperparameters('scaleDj', 3)
        paqu.update_hyperparameters('shapeDj', 0.5)
        paqu.fit(n_iter=GIBBS_ITERS, fit_intercept=True, prior_D='Gaussian', verbose=False)

        Ihat = paqu.I_storer[BURN_IN:, ].mean(axis=0)
        I0hat = paqu.I0_storer[BURN_IN:, ].mean(axis=0)
        Dhat = paqu.D_storer[BURN_IN:, :].mean(axis=0)
        lfsrD = paqu.LFSR('D_storer', BURN_IN)
        time = paqu.fit_time

        row = {
            'block': block,
            'iso': iso_ordered,
            'n': n,
            'q': q,
            'r': r,
            'time': time,
            'Dhat': Dhat.tolist(),
            'LFSR_D': lfsrD.tolist(),
            'Ihat': Ihat.tolist(),
            'I0hat': I0hat.tolist(),
        }

        with open(out_path, 'a') as fd:
            writer = csv.DictWriter(fd, fieldnames=COLUMNS)
            writer.writerow(row)

    Parallel(n_jobs=N_JOBS)(delayed(fx)(block) for block in subset_blocks)
    print(f'Done: {suffix} -> {out_path}')


if __name__ == '__main__':

    Z, assign_block_I, assign_block_P, collapsed_dict = load_annotations_and_blocks()

    T = pd.read_csv('../../T_values.csv', index_col=0)

    with open('real_sim_data/subset_blocks.txt') as fd:
        subset_blocks = [int(b) for b in fd.read().splitlines()]

    if len(sys.argv) == 3:
        replicates = [(int(sys.argv[1]), float(sys.argv[2]))]
    else:
        replicates = list(itertools.product(SEEDS, D_SIZES))

    for seed, D_size in replicates:
        run_replicate(seed, D_size, Z, assign_block_I, collapsed_dict, T, subset_blocks)
