#
# Per-replicate data perturbation for the semi-synthetic real-data DE
# simulation study.
#
# For each (seed, D_size) replicate:
#  - sample 2 * GROUP_SIZE of the 50 real Control samples, split into
#    group0/group1 (GROUP_SIZE vs GROUP_SIZE,
#    A = [0]*GROUP_SIZE + [1]*GROUP_SIZE)
#  - draw ~15% of the subset's 911 collapsed isoforms as "true DE"
#  - shift (additively, in log space) the union of their mapped peptide
#    columns, for group1 samples only, by D_size
#
# Outputs (per replicate, in real_sim_data/):
#   sample_assignment_seed{S}_D{D}{tag}.csv  -- HU.ID, group (0/1), in row order
#   ground_truth_seed{S}_D{D}{tag}.csv       -- block, iso, is_DE, n_mapped_peptides,
#                                                n_shifted_peptides, D_size, seed
#   P_perturbed_seed{S}_D{D}{tag}.csv        -- 2*GROUP_SIZE x ~10,000 perturbed peptide matrix
# where {tag} is '' for GROUP_SIZE == 10 (the original 10 vs 10 study) and
# '_{GROUP_SIZE}v{GROUP_SIZE}' otherwise.
#

import ast

import numpy as np
import pandas as pd

SEEDS = [0]
D_SIZES = [0.33, 0.66, 1]
DE_FRACTION = 0.15
GROUP_SIZE = 5
GROUP_TAG = '' if GROUP_SIZE == 10 else f'_{GROUP_SIZE}v{GROUP_SIZE}'

if __name__ == '__main__':

    Xp = pd.read_csv('../../meta_peptide.csv', index_col=0)
    controls = Xp[Xp.DX == 'C'].index.tolist()
    assert len(controls) == 50, f'expected 50 controls, got {len(controls)}'

    subset_isoforms = pd.read_csv('real_sim_data/subset_isoforms.csv')
    subset_isoforms['mapped_peptides'] = subset_isoforms['mapped_peptides'].apply(ast.literal_eval)

    with open('real_sim_data/subset_peptides.txt') as fd:
        subset_peptides = fd.read().splitlines()

    P_full = pd.read_csv('../../P_values.csv', index_col=0)
    P_subset = P_full.loc[:, subset_peptides]

    n_de = int(round(DE_FRACTION * len(subset_isoforms)))
    print(f'{len(subset_isoforms)} subset isoforms, {n_de} ({DE_FRACTION:.0%}) chosen as DE per replicate')

    for seed in SEEDS:
        for d_idx, D_size in enumerate(D_SIZES):
            rng = np.random.RandomState(seed * 100 + d_idx)

            # GROUP_SIZE vs GROUP_SIZE sample split
            selected = rng.choice(controls, 2 * GROUP_SIZE, replace=False)
            order = selected.tolist()
            group1 = order[GROUP_SIZE:]
            A = np.array([0] * GROUP_SIZE + [1] * GROUP_SIZE)

            # DE isoform set
            de_idx = rng.choice(len(subset_isoforms), n_de, replace=False)
            is_DE = np.zeros(len(subset_isoforms), dtype=int)
            is_DE[de_idx] = 1

            # union of mapped peptides across DE isoforms -> shifted once
            union_peptides = set()
            for i in de_idx:
                union_peptides.update(subset_isoforms.loc[i, 'mapped_peptides'])
            union_peptides = sorted(union_peptides)

            n_shifted = [
                len(set(mp) & set(union_peptides))
                for mp in subset_isoforms['mapped_peptides']
            ]

            gt = subset_isoforms[['block', 'iso']].copy()
            gt['is_DE'] = is_DE
            gt['n_mapped_peptides'] = subset_isoforms['mapped_peptides'].apply(len)
            gt['n_shifted_peptides'] = n_shifted
            gt['D_size'] = D_size
            gt['seed'] = seed

            P_rep = P_subset.loc[order].copy()
            P_rep.loc[group1, union_peptides] += D_size

            suffix = f'seed{seed}_D{D_size}{GROUP_TAG}'
            pd.DataFrame({'HU.ID': order, 'group': A}).set_index('HU.ID').to_csv(
                f'real_sim_data/sample_assignment_{suffix}.csv')
            gt.to_csv(f'real_sim_data/ground_truth_{suffix}.csv', index=False)
            P_rep.to_csv(f'real_sim_data/P_perturbed_{suffix}.csv')

            print(f'seed={seed} D_size={D_size}: {len(union_peptides)} peptides shifted, '
                  f'{is_DE.sum()} DE isoforms')
