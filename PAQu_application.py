#
# PAQu application
#

from PAQu import PAQu

import numpy as np
import pandas as pd
import networkx as nx
import csv

from joblib import Parallel, delayed
import multiprocessing

if __name__ == '__main__':

    np.random.seed(1)

    version = '_collapsed_isoforms_aligned_age6'
    collapsed_isoforms = True
    num_cores = multiprocessing.cpu_count() - 1
    gibbs_iters = 3000 # number of iterations in the Gibbs sampler
    burn_in = 2000 # burn-in period
    # incremental_seed = 2
    insert_covariates = True
    insert_transcripts = True
    insert_intercept = True
    only_tQTL = False
    only_pivotal = False

    # results storer
    columns=['block', 'iso', 'n', 'q', 'r', 'time', 'pihat', 'Dhat', 'LFSR_D', 'What', 'LFSR_W', 'Zhat', 'LFSR_Z', 'Ihat', 'I0hat', 'Bihat', 'LFSR_Bi']
    results = pd.DataFrame(columns=columns)

    # Load data
    Z = pd.read_csv('annotations.csv', index_col=0)
    Z = Z.loc[:, ['PEPTIDE.SEQUENCE', 'TRANSCRIPTS']]
    Z.columns = ['peptide', 'enst']

    if collapsed_isoforms is True:
        # group isoforms by compatible peptides: each row is isoform, one column denoting peptide set
        # if same set, collapse into one "representative" isoform
        l = Z.groupby('enst').peptide.apply(set).reset_index()
        l.columns = ['enst', 'peptide_set']
        l['peptide_set'] = l.peptide_set.apply(lambda x: ','.join(sorted(list(x))))
        # group by peptide set
        l = l.groupby('peptide_set').enst.apply(set).reset_index()
        collapsed_dict = l.enst
        # extend it so that it is index:enst, value: set of collapsed isoforms
        collapsed_dict_final = {}
        for i in range(collapsed_dict.shape[0]):
            for elem in collapsed_dict[i]:
                collapsed_dict_final[elem] = collapsed_dict[i]
        collapsed_dict = collapsed_dict_final

    T = pd.read_csv('T_values.csv', index_col=0) # pd.read_csv('transcript_backin.csv', index_col=0)
    P = pd.read_csv('P_values.csv', index_col=0) # pd.read_csv('peptide_backinDXAGE.csv', index_col=0)
    # T = T.T
    # P = P.T

    P = P.loc[P.index.isin(T.index),:]

    P.sort_index(inplace=True)
    T.sort_index(inplace=True)

    GT = pd.read_csv('transcripts-with-significant-tQTL-2024-08-20.txt').transcripts.tolist()

    Xp = pd.read_csv('meta_peptide.csv', index_col=0)
    Xi = pd.read_csv('meta_transcript.csv', index_col=0)

    # Convert to 0 and 1
    Xp.DX = Xp.DX.map({'C': 0, 'Sz': 1})
    A = Xp.DX.values#.reshape(-1, 1)

    # Format Xi
    Xi.SEX = Xi.SEX.map({'Male': 0, 'Female': 1})
    Xi = Xi.drop(['HU.ID', 'DX', 'INSTITUTION'], axis=1)

    # Format Xp
    plex = pd.get_dummies(Xp.PLEX)
    plex[plex==True] = 1
    plex[plex==False] = 0
    Xp = pd.concat([Xp, plex], axis=1)
    Xp = Xp.drop(['HU.ID', 'PLEX', 'DX'], axis=1)

    age = Xp.AGE
    age.sort_index(inplace=True)

    # # Regress age on intercept + A, take orthogonal residuals
    # projection = np.linalg.lstsq(np.vstack([np.ones(Xi.shape[0]), A]).T, age, rcond=None)[0]
    # age = age - projection[0] - A * projection[1]

    Xi = Xi.values
    Xp = Xp.values
    # ageX = age.values.reshape(-1, 1)

    # Define blocks from Z
    G = nx.Graph()
    G.add_edges_from(Z.to_numpy())
    blocks = list(nx.connected_components(G))

    assign_block = {}
    for i, block in enumerate(blocks):
        for element in block:
            assign_block[element] = i
    assign_block = pd.DataFrame.from_dict(assign_block, orient='index').reset_index()
    assign_block.columns = ['element', 'block']
    n_blocks = assign_block.block.max()

    assign_block_P = assign_block.loc[~assign_block.element.str.contains('ENST0'),:]
    assign_block_I = assign_block.loc[assign_block.element.str.contains('ENST0'),:]
    Z['idx'] = 1

    if only_pivotal:
        u = assign_block_I.block.value_counts() == 1
        u = u.reset_index()
        u.columns = ['block', 'dummy']
        # unique_isoforms = assign_block_I[assign_block_I.block.isin(u[u.dummy].block)].element
        assign_block_I = assign_block_I[assign_block_I.block.isin(u[u.dummy].block)]

    def fx(block):

        np.random.seed(block + incremental_seed)

        iso = assign_block_I[assign_block_I.block == block].element
        if len(iso) == 1 and only_tQTL is True:
            if iso[0] not in GT:
                return None

        print('Processing block:', block)

        # associated_peptides = Z[Z.enst.isin(iso)].peptide.unique()
        # P_block = P[associated_peptides]
        # T_block = T.loc[:, iso]

        mask = Z[Z.enst.isin(iso)].pivot(index='enst', columns='peptide', values='idx').fillna(0)
        
        T_block_local = T.loc[:, mask.index].copy()

        if collapsed_isoforms is True:
            mask.sort_index(axis=0, inplace=True)
            mask_cleaned = mask.drop_duplicates(keep='first')
            dropped = mask.index.difference(mask_cleaned.index)

            for dropped_iso in dropped:
                representative_iso = list(collapsed_dict[dropped_iso])
                preserved = [iso for iso in representative_iso if iso in mask_cleaned.index]
                T_block_local.loc[:, preserved] = T_block_local[representative_iso].mean(axis=1).values.reshape(-1,1)

            mask = mask_cleaned
        
        T_block = T_block_local.loc[:, mask.index]
        P_block = P.loc[:, mask.columns]

        n = T_block.shape[0]
        q = T_block.shape[1]
        r = P_block.shape[1]

        iso_ordered = mask.index.tolist()

        P_block = P_block.values
        T_block = T_block.values
        mask = mask.values

        # # Regress T on intercept + D + age, take orthogonal residuals, add back intercept (i.e. only remove D and age)
        # projection = np.linalg.lstsq(np.vstack([np.ones(n), A, age]).T, T_block, rcond=None)[0]
        # T_block = T_block - projection[0] - np.outer(A, projection[1, :]) - np.outer(age, projection[2, :])
        # T_block = T_block.reshape(n, q)

        # Regress T on intercept, A
        projection = np.linalg.lstsq(np.vstack([np.ones(n), A]).T, T_block, rcond=None)[0]
        T_block = T_block - np.outer(A, projection[1, :])
        T_block = T_block.reshape(n, q)

        # Regress age on intercept, A, T
        age_standard = age.values #/ age.std()
        projection = np.linalg.lstsq(np.vstack([np.ones(n), A, T_block.T]).T, age_standard, rcond=None)[0]
        res_age = age_standard - A * projection[1] - T_block @ projection[2:]
        res_age = res_age.reshape(n, 1)

        # For partial transcripts
        # T_block = np.mean(T_block, axis=0)
        # T_block = np.tile(T_block, (n, 1))

        #
        # Fit PAQu
        #

        if insert_covariates is False:
            Xi = None
            Xp = None
        if insert_transcripts is False:
            T_block = q

        paqu = PAQu(A, mask, P_block, T_block, res_age, None)
        paqu.update_hyperparameters('scaleDj', 3)
        paqu.update_hyperparameters('shapeDj', 0.5)
        paqu.fit(n_iter=gibbs_iters, fit_intercept=insert_intercept, prior_D='Gaussian', verbose=False)

        np.save('./full_results' + version + '/I0_storer_block_' + str(block) + '_' + str(incremental_seed) + '.npy', paqu.I0_storer[burn_in:,])
        np.save('./full_results' + version + '/I_storer_block_' + str(block) + '_' + str(incremental_seed) + '.npy', paqu.I_storer[burn_in:,])
        np.save('./full_results' + version + '/W_storer_block_' + str(block) + '_' + str(incremental_seed) + '.npy', paqu.W_storer[burn_in:,])
        np.save('./full_results' + version + '/D_storer_block_' + str(block) + '_' + str(incremental_seed) + '.npy', paqu.D_storer[burn_in:,:])
        np.save('./full_results' + version + '/Z_storer_block_' + str(block) + '_' + str(incremental_seed) + '.npy', paqu.Z_storer[burn_in:,:])
        np.save('./full_results' + version + '/Bi_storer_block_' + str(block) + '_' + str(incremental_seed) + '.npy', paqu.Bi_storer[burn_in:,])
        
        I0hat = paqu.I0_storer[burn_in:,].mean(axis=0)
        Ihat = paqu.I_storer[burn_in:,].mean(axis=0)
        What = paqu.W_storer[burn_in:,].mean(axis=0)
        pihat = paqu.pi_storer[burn_in:,].mean(axis=0)
        Dhat = paqu.D_storer[burn_in:,:].mean(axis=0)
        Zhat = paqu.Z_storer[burn_in:,:].mean(axis=0)
        Bihat = paqu.Bi_storer[burn_in:,].mean(axis=0)

        # P_block.mean(axis=0)
        # T_block.mean(axis=0)

        lfsrD = paqu.LFSR('D_storer', burn_in)
        lfsrW = paqu.LFSR('W_storer', burn_in)
        lfsrZ = paqu.LFSR('Z_storer', burn_in)
        lfsrBi = paqu.LFSR('Bi_storer', burn_in)
        time = paqu.fit_time

        pihat = pihat.tolist()
        Dhat = Dhat.tolist()
        What = What.tolist()
        Zhat = Zhat.tolist()
        lfsrD = lfsrD.tolist()
        lfsrW = lfsrW.tolist()
        lfsrZ = lfsrZ.tolist()
        iso = iso.tolist()
        Ihat = Ihat.tolist()
        I0hat = I0hat.tolist()
        Bihat = Bihat.tolist()
        lfsrBi = lfsrBi.tolist()
     
        # Store results
        # series_to_export = pd.Series([block, iso_ordered, n, q, r, time, pihat, Dhat, lfsrD, What, lfsrW, Zhat, lfsrZ, Ihat, I0hat])
        # series_to_export.index = columns
        # with open('app_results_PAQu' + version + '_seed_' + str(incremental_seed) + '.csv','a') as fd:
        #     writer = csv.DictWriter(fd, fieldnames=columns)
        #     writer.writerow(series_to_export.to_dict())

        row_to_export = {
            'block': block,
            'iso': iso_ordered, # This maps to the 'iso' column
            'n': n,
            'q': q,
            'r': r,
            'time': time,
            'pihat': pihat,
            'Dhat': Dhat,
            'LFSR_D': lfsrD,
            'What': What,
            'LFSR_W': lfsrW,
            'Zhat': Zhat,
            'LFSR_Z': lfsrZ,
            'Ihat': Ihat,
            'I0hat': I0hat,
            'Bihat': Bihat,
            'LFSR_Bi': lfsrBi
        }

        with open('app_results_PAQu' + version + '_seed_' + str(incremental_seed) + '.csv','a') as fd:
            writer = csv.DictWriter(fd, fieldnames=columns)
            writer.writerow(row_to_export)

    for incremental_seed in range(0, 10):
        with open('app_results_PAQu' + version + '_seed_' + str(incremental_seed) + '.csv','a') as fd:
            writer = csv.writer(fd)
            writer.writerow(columns)
        Parallel(n_jobs=num_cores)(delayed(fx)(block) for block in assign_block_I.block.unique().tolist())


