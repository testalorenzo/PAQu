#
# Shared block / mask / collapsed-isoform logic, factored out of
# ../PAQu_app.py (lines 36-115) so that the data-perturbation, PAQu-fitting,
# IsoBayes-fitting and peptide-baseline scripts in this directory all agree
# on the same block definitions.
#
# PAQu_app.py itself is not modified.
#

import pandas as pd
import networkx as nx


def load_annotations_and_blocks(annotations_path='../../annotations.csv'):
    """Load annotations.csv and build the peptide<->isoform bipartite graph.

    Returns
    -------
    Z : DataFrame with columns ['peptide', 'enst', 'idx'] (idx is always 1)
    assign_block_I : DataFrame with columns ['element', 'block'] -- isoforms (ENST)
    assign_block_P : DataFrame with columns ['element', 'block'] -- peptides
    collapsed_dict : dict mapping enst -> set of collapsed-representative ensts
        (isoforms with an identical mapped-peptide set)
    """
    Z = pd.read_csv(annotations_path, index_col=0)
    Z = Z.loc[:, ['PEPTIDE.SEQUENCE', 'TRANSCRIPTS']]
    Z.columns = ['peptide', 'enst']

    # collapsed_dict: isoforms with identical peptide sets are collapsed to
    # one representative (PAQu_app.py lines 40-54)
    l = Z.groupby('enst').peptide.apply(set).reset_index()
    l.columns = ['enst', 'peptide_set']
    l['peptide_set'] = l.peptide_set.apply(lambda x: ','.join(sorted(list(x))))
    l = l.groupby('peptide_set').enst.apply(set).reset_index()
    collapsed_dict = {}
    for i in range(l.shape[0]):
        for elem in l.enst[i]:
            collapsed_dict[elem] = l.enst[i]

    G = nx.Graph()
    G.add_edges_from(Z[['peptide', 'enst']].to_numpy())
    blocks = list(nx.connected_components(G))

    # nx.connected_components iterates over Python sets internally, whose
    # order depends on the process's hash seed -- sort blocks into a
    # canonical, deterministic order so that block IDs are stable across
    # separate script invocations (subset selection, PAQu fitting, IsoBayes
    # fitting, etc. each call this function in their own process).
    blocks.sort(key=lambda block: min(block))

    assign_block = {}
    for i, block in enumerate(blocks):
        for element in block:
            assign_block[element] = i
    assign_block = pd.DataFrame.from_dict(assign_block, orient='index').reset_index()
    assign_block.columns = ['element', 'block']

    assign_block_P = assign_block.loc[~assign_block.element.str.contains('ENST0'), :]
    assign_block_I = assign_block.loc[assign_block.element.str.contains('ENST0'), :]
    Z['idx'] = 1

    return Z, assign_block_I, assign_block_P, collapsed_dict


def get_block_mask(Z, assign_block_I, block, collapsed_dict, collapsed_isoforms=True):
    """Build the isoform x peptide incidence mask for a single block.

    Mirrors PAQu_app.py lines 121, 132, 136-148 (minus the T_block handling,
    which callers do separately since they may need pre-collapse T columns).

    Returns
    -------
    mask : DataFrame, index = collapsed-representative ENSTs, columns = peptide names
    iso_ordered : list of collapsed-representative ENSTs (== mask.index.tolist())
    dropped : list of ENSTs that were collapsed away (duplicates of a representative)
    """
    iso = assign_block_I[assign_block_I.block == block].element
    mask = Z[Z.enst.isin(iso)].pivot(index='enst', columns='peptide', values='idx').fillna(0)

    if collapsed_isoforms:
        mask.sort_index(axis=0, inplace=True)
        mask_cleaned = mask.drop_duplicates(keep='first')
        dropped = mask.index.difference(mask_cleaned.index)
        mask = mask_cleaned
    else:
        dropped = pd.Index([])

    return mask, mask.index.tolist(), dropped.tolist()


def get_block_mask_full(Z, assign_block_I, block):
    """Pre-collapse isoform x peptide incidence mask for a single block
    (used by the IsoBayes fitting script, which needs the true, uncollapsed
    incidence structure for its EC strings / tpm table).

    Returns
    -------
    mask_full : DataFrame, index = ALL original ENSTs in the block,
        columns = peptide names
    """
    iso = assign_block_I[assign_block_I.block == block].element
    mask_full = Z[Z.enst.isin(iso)].pivot(index='enst', columns='peptide', values='idx').fillna(0)
    return mask_full
