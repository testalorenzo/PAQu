#
# IsoBayes fitting on the 20-sample (10 vs 10), ~10k-peptide perturbed
# subset of the real dataset (one run per (seed, D_size) replicate).
#
# IsoBayes is fit separately to each of the 20 samples, but each fit sees
# the ENTIRE ~10k-peptide subset peptidome in one inference() call (per
# user correction -- it is not run per-block). The EC strings use the true,
# pre-collapse ENST incidence structure (get_block_mask_full); the resulting
# n x q_original Ihat matrix is then collapsed to match PAQu's
# collapsed-isoform representatives by averaging columns within each
# collapsed_dict group, mirroring PAQu_app.py:141-144.
#
# Per-sample inference() calls are parallelized via run_isobayes_persample.R
# (one sample per call), run in parallel across samples with joblib.
#
# Usage:
#   python PAQu_simulations_isobayes_real.py                  # all 9 replicates, all 20 samples
#   python PAQu_simulations_isobayes_real.py <seed> <D_size>            # 1 replicate, all 20 samples
#   python PAQu_simulations_isobayes_real.py <seed> <D_size> <n_samples> # 1 replicate, first n_samples (smoke test)
#   ... --exp                                  # exponentiate (2**x) P/T before
#                                               # building IsoBayes's Y/tpm inputs.
#                                               # P_values.csv/T_values.csv are
#                                               # log2-scale; IsoBayes expects
#                                               # PSM counts / linear-scale TPM
#                                               # (per run_isobayes_persample.R /
#                                               # isobayes_simulation_replication).
#                                               # PAQu's own input is unaffected.
#
# Output (per replicate): sim_results_isobayes_real_seed{S}_D{D}.csv
# (or sim_results_isobayes_real_exp_seed{S}_D{D}.csv with --exp)
#   columns: block, iso, n, q, r, time, Ihat, pval_isobayes
#

import os
import sys
import csv
import time
import tempfile
import itertools
import subprocess

import numpy as np
import pandas as pd
import scipy.stats as stats

from joblib import Parallel, delayed

from _blocks import load_annotations_and_blocks, get_block_mask, get_block_mask_full

SEEDS = [0]
D_SIZES = [0.33, 0.66, 1]
GROUP_SIZE = 5
GROUP_TAG = '' if GROUP_SIZE == 10 else f'_{GROUP_SIZE}v{GROUP_SIZE}'
R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_isobayes_persample.R')
DE_R_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_de_methods.R')

COLUMNS = ['block', 'iso', 'n', 'q', 'r', 'time', 'Ihat', 'pval_isobayes',
           'pval_edger', 'pval_deseq2', 'pval_deseq2_sf1', 'pval_limma', 'pval_aldex2']


def build_block_info(Z, assign_block_I, collapsed_dict, subset_blocks):
    """For all subset_blocks, build:
      - ec_map: peptide -> pipe-separated ORIGINAL (pre-collapse) ENST ids
      - all_original_isoforms: list of all original ENSTs across subset_blocks
      - block_groups: dict block -> dict {collapsed_rep: [original ENSTs in its collapsed group]}
      - block_r: dict block -> number of peptides in the block
    """
    ec_map = {}
    all_original_isoforms = []
    block_groups = {}
    block_r = {}

    for block in subset_blocks:
        mask_full = get_block_mask_full(Z, assign_block_I, block)
        for p in mask_full.columns:
            ec_map[p] = '|'.join(mask_full.index[mask_full[p] == 1].tolist())
        all_original_isoforms.extend(mask_full.index.tolist())
        block_r[block] = mask_full.shape[1]

        mask, iso_ordered, dropped = get_block_mask(Z, assign_block_I, block, collapsed_dict)
        groups = {}
        for rep in iso_ordered:
            groups[rep] = sorted(set(collapsed_dict[rep]) & set(mask_full.index))
        block_groups[block] = groups

    return ec_map, all_original_isoforms, block_groups, block_r


def run_isobayes_one_sample(k, sample_id, P_row, T_row, ec_map, subset_peptides, all_original_isoforms,
                             exp_transform=False):
    """Run IsoBayes on the whole subset peptidome for a single sample."""

    Y = P_row[subset_peptides].values
    tpm = T_row[all_original_isoforms].values
    if exp_transform:
        # P_values.csv / T_values.csv are log2-scale; IsoBayes's "psm" Y
        # input expects PSM counts and its tpm prior expects linear-scale
        # TPM, so undo the log2 transform before handing off.
        Y = np.round(np.exp2(Y)).astype(int)
        tpm = np.exp2(tpm)

    peptides_df = pd.DataFrame({
        'sample': k,
        'Y': Y,
        'EC': [ec_map[p] for p in subset_peptides],
    })
    tpm_df = pd.DataFrame({
        'sample': k,
        'isoname': all_original_isoforms,
        'tpm': tpm,
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        pep_csv = os.path.join(tmpdir, 'peptides.csv')
        tpm_csv = os.path.join(tmpdir, 'tpm.csv')
        out_csv = os.path.join(tmpdir, 'isoform_results.csv')
        peptides_df.to_csv(pep_csv, index=False)
        tpm_df.to_csv(tpm_csv, index=False)

        start = time.time()
        subprocess.run(
            ['Rscript', R_SCRIPT, pep_csv, tpm_csv, out_csv],
            check=True, capture_output=True, text=True
        )
        elapsed = time.time() - start

        out = pd.read_csv(out_csv)

    return out, elapsed


def run_replicate(seed, D_size, Z, assign_block_I, collapsed_dict, T, subset_blocks,
                   ec_map, all_original_isoforms, block_groups, block_r, n_samples=2 * GROUP_SIZE, n_jobs=20,
                   exp_transform=False):

    suffix = f'seed{seed}_D{D_size}{GROUP_TAG}'
    out_suffix = f'exp_{suffix}' if exp_transform else suffix

    sample_assignment = pd.read_csv(f'real_sim_data/sample_assignment_{suffix}.csv', index_col=0)
    order = sample_assignment.index.tolist()
    A = sample_assignment['group'].values

    P = pd.read_csv(f'real_sim_data/P_perturbed_{suffix}.csv', index_col=0).loc[order]
    T_rep = T.loc[order]

    subset_peptides = P.columns.tolist()

    samples_to_run = list(enumerate(order))[:n_samples]

    results = Parallel(n_jobs=min(n_jobs, len(samples_to_run)))(
        delayed(run_isobayes_one_sample)(
            k, sample_id, P.loc[sample_id], T_rep.loc[sample_id], ec_map, subset_peptides, all_original_isoforms,
            exp_transform=exp_transform
        )
        for k, sample_id in samples_to_run
    )

    outs, elapsed_list = zip(*results)
    out = pd.concat(outs, ignore_index=True)
    total_elapsed = sum(elapsed_list)

    if n_samples < 2 * GROUP_SIZE:
        print(f'{suffix}: ran {n_samples} sample(s), per-sample wall time = {elapsed_list}')
        return

    Ihat_full = out.pivot(index='sample', columns='Isoform', values='Abundance').sort_index()
    Ihat_full = Ihat_full[all_original_isoforms]  # n x q_original, rows in `order`

    # As suggested by the IsoBayes authors, also run edgeR (QL F-test),
    # DESeq2 (Wald test) and limma-voom on a counts matrix obtained by
    # rounding the per-sample IsoBayes Abundance estimates, comparing
    # group A == 0 vs A == 1 -- once for the whole subset peptidome.
    counts = np.maximum(np.round(Ihat_full.values), 0).astype(int)  # n x q_original
    with tempfile.TemporaryDirectory() as tmpdir:
        counts_csv = os.path.join(tmpdir, 'counts.csv')
        group_csv = os.path.join(tmpdir, 'group.csv')
        de_out_csv = os.path.join(tmpdir, 'de_results.csv')

        counts_df = pd.DataFrame(counts.T, columns=[f'sample{k}' for k in range(len(order))])
        counts_df.insert(0, 'isoform', all_original_isoforms)
        counts_df.to_csv(counts_csv, index=False)
        pd.DataFrame({'group': A.astype(int)}).to_csv(group_csv, index=False)

        subprocess.run(
            ['Rscript', DE_R_SCRIPT, counts_csv, group_csv, de_out_csv],
            check=True, capture_output=True, text=True
        )

        de_out = pd.read_csv(de_out_csv).set_index('isoform')

    out_path = f'sim_results_isobayes_real_{out_suffix}.csv'
    with open(out_path, 'w') as fd:
        writer = csv.writer(fd)
        writer.writerow(COLUMNS)

    for block in subset_blocks:
        groups = block_groups[block]
        iso_ordered = list(groups.keys())
        if len(iso_ordered) == 0:
            continue

        Ihat_block = np.column_stack([Ihat_full[groups[rep]].mean(axis=1).values for rep in iso_ordered])
        pvals = [
            stats.ttest_ind(Ihat_block[A == 0, i], Ihat_block[A == 1, i]).pvalue
            for i in range(len(iso_ordered))
        ]

        row = {
            'block': block,
            'iso': iso_ordered,
            'n': Ihat_block.shape[0],
            'q': Ihat_block.shape[1],
            'r': block_r[block],
            'time': total_elapsed,
            'Ihat': Ihat_block.tolist(),
            'pval_isobayes': pvals,
            'pval_edger': de_out.loc[iso_ordered, 'pval_edger'].tolist(),
            'pval_deseq2': de_out.loc[iso_ordered, 'pval_deseq2'].tolist(),
            'pval_deseq2_sf1': de_out.loc[iso_ordered, 'pval_deseq2_sf1'].tolist(),
            'pval_limma': de_out.loc[iso_ordered, 'pval_limma'].tolist(),
            'pval_aldex2': de_out.loc[iso_ordered, 'pval_aldex2'].tolist(),
        }

        with open(out_path, 'a') as fd:
            writer = csv.DictWriter(fd, fieldnames=COLUMNS)
            writer.writerow(row)

    print(f'Done: {out_suffix} -> {out_path} (total IsoBayes wall time: {total_elapsed:.1f}s)')


if __name__ == '__main__':

    Z, assign_block_I, assign_block_P, collapsed_dict = load_annotations_and_blocks()
    T = pd.read_csv('../../T_values.csv', index_col=0)

    with open('real_sim_data/subset_blocks.txt') as fd:
        subset_blocks = [int(b) for b in fd.read().splitlines()]

    ec_map, all_original_isoforms, block_groups, block_r = build_block_info(
        Z, assign_block_I, collapsed_dict, subset_blocks)

    print(f'{len(all_original_isoforms)} original isoforms, {len(ec_map)} peptides in subset')

    exp_transform = '--exp' in sys.argv
    argv = [a for a in sys.argv[1:] if a != '--exp']

    if len(argv) >= 2:
        n_samples = int(argv[2]) if len(argv) >= 3 else 2 * GROUP_SIZE
        replicates = [(int(argv[0]), float(argv[1]), n_samples)]
    else:
        replicates = [(s, d, 2 * GROUP_SIZE) for s, d in itertools.product(SEEDS, D_SIZES)]

    for seed, D_size, n_samples in replicates:
        run_replicate(seed, D_size, Z, assign_block_I, collapsed_dict, T, subset_blocks,
                       ec_map, all_original_isoforms, block_groups, block_r, n_samples=n_samples,
                       exp_transform=exp_transform)
