#
# Format data for upload to shared folder
#

import pandas as pd
import numpy as np

def collapse_match(iso, collapsed_dict, already_clean=False):
    if already_clean:
        iso_clean = iso
        return '+'.join(sorted(list(collapsed_dict[iso_clean])))
    else:
        iso_clean = [eval(x) for x in iso]
        return ['+'.join(sorted(list(collapsed_dict[y]))) for y in iso_clean]

if __name__ == '__main__':

    pos = 'collapsed_isoforms_aligned_age6_' # 'truncated_prior_aligned_new_' # 'collapsed_isoforms_aligned_age_', 'collapsed_isoforms_aligned_new_
    collapsed_isoforms = True

    if collapsed_isoforms is True:
        # Load data
        Z = pd.read_csv('annotations.csv', index_col=0)
        Z = Z.loc[:, ['PEPTIDE.SEQUENCE', 'TRANSCRIPTS']]
        Z.columns = ['peptide', 'enst']

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

    #
    # D
    #

    print('Processing D_hat...')

    results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_0.csv')
    results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
    results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
    results_full['LFSR_D'] = results_full['LFSR_D'].apply(lambda x: x[1:-1].split(','))
    results_full['Dhat_0'] = results_full['Dhat'].apply(lambda x: x[1:-1].split(','))
    
    if collapsed_isoforms is False:
        results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

    results_full = results_full.explode(['iso', 'LFSR_D', 'Dhat_0'])
    # results_full.iso = results_full.iso.apply(eval)
 
    results_full['LFSR_D_0'] = results_full['LFSR_D'].astype(float)
    final = results_full[['block', 'iso', 'LFSR_D_0', 'Dhat_0', 'r', 'q']]

    for dataset in range(1, 10):
        results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_' + str(dataset) + '.csv')
        results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
        results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
        results_full['LFSR_D'] = results_full['LFSR_D'].apply(lambda x: x[1:-1].split(','))
        results_full['Dhat'] = results_full['Dhat'].apply(lambda x: x[1:-1].split(','))
        if collapsed_isoforms is False:
            results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

        results_full = results_full.explode(['iso','LFSR_D', 'Dhat'])
        # results_full.iso = results_full.iso.apply(eval)

        results_full['LFSR_D'] = results_full['LFSR_D'].astype(float)
        results_full = results_full[['iso', 'Dhat', 'LFSR_D']]
        results_full.columns = ['iso', 'Dhat_' + str(dataset), 'LFSR_D_' + str(dataset)]
        final = pd.merge(final, results_full, on=['iso'])

    # Filter out rows with q > r
    final = final[final.q <= final.r]

    # Order columns alphabetically
    final = final.drop(columns=['q', 'r'])
    final = final.reindex(sorted(final.columns), axis=1)

    # block and iso first
    final = final[['block', 'iso'] + [col for col in final.columns if col not in ['block', 'iso']]]

    # Export
    final.to_csv('D_hat' + pos + '.csv', index=False)

    #
    # W
    #

    print('Processing W_hat...')

    results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_0.csv')
    results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
    results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
    results_full['LFSR_W_0'] = results_full['LFSR_W'].apply(lambda x: x[1:-1].split(','))
    results_full['What_0'] = results_full['What'].apply(lambda x: x[1:-1].split(','))
    
    if collapsed_isoforms is False:
        results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

    results_full = results_full.explode(['iso', 'LFSR_W_0', 'What_0'])
    # results_full.iso = results_full.iso.apply(eval)
 
    final = results_full[['block', 'iso', 'LFSR_W_0', 'What_0', 'r', 'q']]

    for dataset in range(1, 10):
        results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_' + str(dataset) + '.csv')
        results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
        results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
        results_full['LFSR_W'] = results_full['LFSR_W'].apply(lambda x: x[1:-1].split(','))
        results_full['What'] = results_full['What'].apply(lambda x: x[1:-1].split(','))
        
        if collapsed_isoforms is False:
            results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

        results_full = results_full.explode(['iso', 'LFSR_W', 'What'])
        # results_full.iso = results_full.iso.apply(eval)

        results_full = results_full[['iso', 'LFSR_W', 'What']]
        results_full.columns = ['iso', 'LFSR_W_' + str(dataset), 'What_' + str(dataset)]
        final = pd.merge(final, results_full, on=['iso'], suffixes=(None, '_'+str(dataset)))

    # Filter out rows with q > r
    final = final[final.q <= final.r]

    # Order columns: block, iso, Whats, and LFSR_Ws
    final = final.drop(columns=['q', 'r'])
    final = final.reindex(sorted(final.columns), axis=1)
    final = final[['block', 'iso'] + [col for col in final.columns if 'What' in col] + [col for col in final.columns if 'LFSR_W' in col]]

    # Export
    final.to_csv('W_hat' + pos + '.csv', index=False)

    #
    # Z
    #

    print('Processing Z_hat...')

    Z = pd.read_csv('annotations.csv', index_col=0)
    Z = Z.loc[:, ['PEPTIDE.SEQUENCE', 'TRANSCRIPTS']]
    Z.columns = ['peptide', 'enst']
    
    results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_0.csv')
    results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
    results_full['Zhat_0'] = results_full['Zhat'].apply(eval)
    results_full['LFSR_Z_0'] = results_full['LFSR_Z'].apply(eval)

    results_full = results_full[results_full.q <= results_full.r]

    final = results_full[['block', 'iso', 'Zhat_0', 'LFSR_Z_0']]
    final.iso = final.iso.apply(lambda x: [eval(y) for y in x])
    final = final.explode(['iso', 'Zhat_0', 'LFSR_Z_0'])

    # Drop zero in lists in Zhat_1 and corresponding LFSR_Z_1
    final['masked'] = final.Zhat_0.apply(lambda x: [y != 0 for y in x])
    #final.Zhat_1 = final.Zhat_1.apply(lambda x: [i for i in x if i != 0])
    final.Zhat_0 = final.apply(lambda x: [x['Zhat_0'][i] for i in range(len(x['Zhat_0'])) if x['masked'][i]], axis=1)
    final.LFSR_Z_0 = final.apply(lambda x: [x['LFSR_Z_0'][i] for i in range(len(x['LFSR_Z_0'])) if x['masked'][i]], axis=1)

    final = final[['block', 'iso', 'Zhat_0', 'LFSR_Z_0']]

    # Attach all other datasets here
    for dataset in range(1, 10):
        results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_' + str(dataset) + '.csv')
        results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
        results_full['Zhat'] = results_full['Zhat'].apply(eval)
        results_full['LFSR_Z'] = results_full['LFSR_Z'].apply(eval)
        
        results_full = results_full[results_full.q <= results_full.r]

        results_full = results_full[['iso', 'Zhat', 'LFSR_Z']]
        results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])
        results_full = results_full.explode(['iso', 'Zhat', 'LFSR_Z'])

        # Drop zero in lists in Zhat and corresponding LFSR_Z
        results_full['masked'] = results_full.Zhat.apply(lambda x: [y != 0 for y in x])
        results_full.Zhat = results_full.apply(lambda x: [x['Zhat'][i] for i in range(len(x['Zhat'])) if x['masked'][i]], axis=1)
        results_full.LFSR_Z = results_full.apply(lambda x: [x['LFSR_Z'][i] for i in range(len(x['LFSR_Z'])) if x['masked'][i]], axis=1)
        # results_full.Zhat = results_full.Zhat.apply(lambda x: [i for i in x if i != 0])
        results_full = results_full[['iso', 'Zhat', 'LFSR_Z']]
        results_full.columns = ['iso', 'Zhat_' + str(dataset), 'LFSR_Z_' + str(dataset)]
        final = pd.merge(final, results_full, on=['iso'])

    for iso in final.iso:
        lst = Z[Z.enst.isin([iso])].peptide.sort_values().values.tolist()
        final.loc[final.iso==iso, 'peptide'] = str(lst)

    final.peptide = final.peptide.apply(eval)
    final2 = final.explode(['peptide'] + [col for col in final.columns if 'Zhat' in col] + [col for col in final.columns if 'LFSR_Z' in col])

    # Order columns: block, iso, peptide, Zhat, and LFSR_Z
    final2 = final2[['block', 'iso', 'peptide'] + [col for col in final2.columns if 'Zhat' in col] + [col for col in final2.columns if 'LFSR_Z' in col]]
    final2['iso'] = final2['iso'].apply(lambda x: collapse_match(x, collapsed_dict, already_clean=True) if collapsed_isoforms else x)


    # Export
    final2.to_csv('Z_hat' + pos + '.csv', index=False)

    #
    # I
    #

    print('Processing I_hat...')

    # output: block,iso,subject x seed
    results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_0.csv')
    results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
    results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
    results_full['Ihat_0'] = results_full['Ihat'].apply(eval)
    results_full = results_full[results_full.q <= results_full.r]
    
    if collapsed_isoforms is False:
        results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

    idx = pd.read_csv('T_values.csv', index_col=0).sort_index().index.tolist()

    results_full['idx'] = [idx for i in range(len(results_full))]
    results_full = results_full.explode(['Ihat_0', 'idx'])

    final = results_full[['block', 'iso', 'idx', 'Ihat_0']]
    final = final.explode(['iso', 'Ihat_0'])

    for dataset in range(1, 10):
        results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_' + str(dataset) + '.csv')
        results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
        results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)    
        results_full['Ihat'] = results_full['Ihat'].apply(eval)
        results_full = results_full[results_full.q <= results_full.r]
        
        if collapsed_isoforms is False:
            results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

        results_full['idx'] = [idx for i in range(len(results_full))]
        results_full = results_full.explode(['Ihat', 'idx']).explode(['iso', 'Ihat'])

        results_full = results_full[['iso', 'idx', 'Ihat']]
        results_full.columns = ['iso', 'idx', 'Ihat_' + str(dataset)]
        final = pd.merge(final, results_full, on=['iso', 'idx'])

    final = final.reindex(sorted(final.columns), axis=1)
    final = final[['block', 'iso', 'idx'] + [col for col in final.columns if 'Ihat' in col]]

    # Export
    final.to_csv('I_hat' + pos + '.csv', index=False)

    #
    # I0
    #

    print('Processing I0_hat...')

    results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_0.csv')
    results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
    results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
    results_full['I0hat_0'] = results_full['I0hat'].apply(lambda x: x[1:-1].split(','))
    
    if collapsed_isoforms is False:
        results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

    results_full = results_full.explode(['iso', 'I0hat_0'])
    # results_full.iso = results_full.iso.apply(eval)
 
    final = results_full[['block', 'iso', 'I0hat_0', 'r', 'q']]

    for dataset in range(1, 10):
        results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_' + str(dataset) + '.csv')
        results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
        results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
        results_full['I0hat'] = results_full['I0hat'].apply(lambda x: x[1:-1].split(','))
        
        if collapsed_isoforms is False:
            results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

        results_full = results_full.explode(['iso', 'I0hat'])
        # results_full.iso = results_full.iso.apply(eval)

        results_full = results_full[['iso', 'I0hat']]
        results_full.columns = ['iso', 'I0hat_' + str(dataset)]
        final = pd.merge(final, results_full, on=['iso'])

    # Filter out rows with q > r
    final = final[final.q <= final.r]

    # Order columns alphabetically
    final = final.drop(columns=['q', 'r'])
    final = final.reindex(sorted(final.columns), axis=1)

    # block and iso first
    final = final[['block', 'iso'] + [col for col in final.columns if col not in ['block', 'iso']]]

    # Export
    final.to_csv('I0_hat' + pos + '.csv', index=False)

    #
    # BI
    #

    print('Processing BI_hat...')

    results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_0.csv')
    results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
    results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
    results_full['LFSR_Bi'] = results_full['LFSR_Bi'].apply(lambda x: x[2:-2].split(','))
    results_full['Bihat_0'] = results_full['Bihat'].apply(lambda x: x[2:-2].split(','))
    
    if collapsed_isoforms is False:
        results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

    results_full = results_full.explode(['iso', 'LFSR_Bi', 'Bihat_0'])
    # results_full.iso = results_full.iso.apply(eval)
 
    results_full['LFSR_Bi_0'] = results_full['LFSR_Bi'].astype(float)
    final = results_full[['block', 'iso', 'LFSR_Bi_0', 'Bihat_0', 'r', 'q']]
    
    for dataset in range(1, 10):
        results_full = pd.read_csv('app_results_PAQu_' + pos + 'seed_' + str(dataset) + '.csv')
        results_full['iso'] = results_full['iso'].apply(lambda x: x[1:-1].split(','))
        results_full['iso'] = results_full['iso'].apply(lambda x: collapse_match(x, collapsed_dict) if collapsed_isoforms else x)
        results_full['LFSR_Bi'] = results_full['LFSR_Bi'].apply(lambda x: x[2:-2].split(','))
        results_full['Bihat'] = results_full['Bihat'].apply(lambda x: x[2:-2].split(','))
        if collapsed_isoforms is False:
            results_full.iso = results_full.iso.apply(lambda x: [eval(y) for y in x])

        results_full = results_full.explode(['iso','LFSR_Bi', 'Bihat'])
        # results_full.iso = results_full.iso.apply(eval)

        results_full['LFSR_Bi'] = results_full['LFSR_Bi'].astype(float)
        results_full = results_full[['iso', 'Bihat', 'LFSR_Bi']]
        results_full.columns = ['iso', 'Bihat_' + str(dataset), 'LFSR_Bi_' + str(dataset)]
        final = pd.merge(final, results_full, on=['iso'])

    # Filter out rows with q > r
    final = final[final.q <= final.r]

    # Order columns alphabetically
    final = final.drop(columns=['q', 'r'])
    final = final.reindex(sorted(final.columns), axis=1)

    # block and iso first
    final = final[['block', 'iso'] + [col for col in final.columns if col not in ['block', 'iso']]]

    # Export
    final.to_csv('BI_hat' + pos + '.csv', index=False)