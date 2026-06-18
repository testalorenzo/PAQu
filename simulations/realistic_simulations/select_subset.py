#
# One-time selection of a ~10,000-peptide subset of the real dataset
# (P_values.csv / annotations.csv), used as the "anchor" for the
# semi-synthetic real-data DE simulation study (reviewer-requested
# 10-vs-10, real-transcriptome simulation).
#
# The sampling unit is the BLOCK (connected component of the
# peptide<->isoform bipartite graph from annotations.csv), never an
# individual peptide: every peptide belonging to a sampled block is included
# in full, so no isoform's mapped-peptide set is ever split across the
# subset boundary.
#
# Outputs (this directory):
#   real_sim_data/subset_blocks.txt    -- one block id per line
#   real_sim_data/subset_peptides.txt  -- one peptide column name per line
#   real_sim_data/subset_isoforms.csv  -- columns: block, iso, mapped_peptides
#                                          (mapped_peptides = repr() of a list
#                                          of peptide column names, one row
#                                          per collapsed isoform)
#

import numpy as np
import pandas as pd

from _blocks import load_annotations_and_blocks, get_block_mask

TARGET_N_PEPTIDES = 10_000

if __name__ == '__main__':

    Z, assign_block_I, assign_block_P, collapsed_dict = load_annotations_and_blocks()

    all_blocks = sorted(assign_block_I.block.unique().tolist())

    # Random ordering of whole blocks
    rng = np.random.RandomState(0)
    block_order = rng.permutation(all_blocks)

    subset_blocks = []
    subset_peptides = []
    seen_peptides = set()
    skipped_collapsed = 0

    for block in block_order:
        # Skip blocks where PAQu would collapse two-or-more isoforms with
        # identical mapped-peptide sets into one representative -- keeps a
        # clean 1:1 correspondence between PAQu isoforms and original ENSTs
        # (no IsoBayes Ihat-averaging needed downstream).
        _, _, dropped = get_block_mask(Z, assign_block_I, block, collapsed_dict)
        if len(dropped) > 0:
            skipped_collapsed += 1
            continue

        peptides = assign_block_P.loc[assign_block_P.block == block, 'element'].tolist()
        new_peptides = [p for p in peptides if p not in seen_peptides]

        subset_blocks.append(int(block))
        subset_peptides.extend(new_peptides)
        seen_peptides.update(new_peptides)

        if len(subset_peptides) >= TARGET_N_PEPTIDES:
            break

    print(f'Selected {len(subset_blocks)} blocks, {len(subset_peptides)} peptides '
          f'(skipped {skipped_collapsed} blocks with collapsed isoforms)')

    # Build subset_isoforms: one row per collapsed isoform in subset_blocks
    rows = []
    for block in subset_blocks:
        mask, iso_ordered, dropped = get_block_mask(Z, assign_block_I, block, collapsed_dict)
        for iso_name in iso_ordered:
            mapped_peptides = mask.columns[mask.loc[iso_name] == 1].tolist()
            rows.append({
                'block': block,
                'iso': iso_name,
                'mapped_peptides': repr(mapped_peptides),
            })

    subset_isoforms = pd.DataFrame(rows)
    print(f'Selected {len(subset_isoforms)} collapsed isoforms')

    with open('real_sim_data/subset_blocks.txt', 'w') as fd:
        fd.write('\n'.join(str(b) for b in subset_blocks))

    with open('real_sim_data/subset_peptides.txt', 'w') as fd:
        fd.write('\n'.join(subset_peptides))

    subset_isoforms.to_csv('real_sim_data/subset_isoforms.csv', index=False)
