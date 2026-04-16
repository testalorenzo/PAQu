#
# Real Simulation Scenarios
#

import pandas as pd
import numpy as np
import itertools
import csv
import scipy.stats as stats

from PAQu import PAQu

from joblib import Parallel, delayed
import multiprocessing

def fx(t):
    seed, n, (q, r, D_active), D_size, D_prior = t
    print('Seed:', seed, 'Sample size:', n, 'q:', q, 'r:', r, '#_active:', D_active, 'Size_active:', D_size, 'D_prior:', D_prior)

    # Set seed
    np.random.seed(seed)

    # Generate data
    if active_intercept is True:
        I0 = np.outer(np.ones(n), np.random.normal(10, 0.5, q))
    else:
        I0 = np.zeros((n, q))
    T = np.random.normal(3, 1, (n, q))
    W = np.random.normal(1, 0.5, q)
    W = np.diag(W)
    if active_transcripts is False:
        W = np.zeros((q, q))
    A = np.ones(n)
    A[0:n//2] = 0
    D = np.zeros(q)
    D[0:D_active] = D_size
    E = np.random.normal(0, 1, (n, q))
    I = I0 + T @ W + np.outer(A, D) + E
    # Z = np.abs(np.random.normal(1, 1, (q, r)))
    a, b = (0.5 - 1) / np.sqrt(1), (1.5 - 1) / np.sqrt(1)
    Z = stats.truncnorm.rvs(a, b, loc=1, scale=np.sqrt(1), size=(q, r))
    if q == 1:
        mask = np.random.binomial(1, 1, (q, r))
    else:
        mask = np.random.binomial(1, 0.3, size=(q, r)) # make sure this is not too sparse
        check = sum(mask) == 0
        for i in range(r):
            if check[i]:
                mask[np.random.choice(q), i] = 1
        for i in range(q):
            if sum(mask[i,]) == 0:
                mask[i, np.random.choice(r)] = 1
    
    Z = Z * mask
    F = np.random.normal(0, 1, (n, r))
    P = I @ Z + F

    #
    # Fit PAQu
    #
    
    paqu = PAQu(A, mask, P, T)
    paqu.fit(n_iter=gibbs_iters, fit_intercept=active_intercept, prior_D=D_prior, verbose=False)

    Ihat = paqu.I_storer[burn_in:,].mean(axis=0)
    What = paqu.W_storer[burn_in:,].mean(axis=0)
    Zhat = paqu.Z_storer[burn_in:,].mean(axis=0)
    pihat = paqu.pi_storer[burn_in:,].mean(axis=0)
    Dhat = paqu.D_storer[burn_in:,:].mean(axis=0)
    Dhat_debiased = (paqu.D_storer[burn_in:,:] * paqu.pi_storer[burn_in:,])
    Dhat_debiased[Dhat_debiased == 0] = np.nan
    Dhat_debiased = np.nanmean(Dhat_debiased, axis=0)

    I_avg = I.mean(axis=0)
    Ihat_avg = Ihat.mean(axis=0)

    Phat = Ihat @ Zhat

    IIhat = np.concatenate([I, Ihat], axis=1)
    cors = np.corrcoef(IIhat.T)
    c = []
    for i in range(q):
        c.append(cors[i, q+i])

    lfsr = paqu.LFSR('D_storer', burn_in)
    lfsrW = paqu.LFSR('W_storer', burn_in)
    time = paqu.fit_time

    pihat = pihat.tolist()
    Dhat = Dhat.tolist()
    Dhat_debiased = Dhat_debiased.tolist()
    lfsr = lfsr.tolist()
    What = What.tolist()
    lfsrW = lfsrW.tolist()

    # Store results
    series_to_export = pd.Series([seed, n, q, r, D_active, D_size, D_prior, time, c, pihat, Dhat, Dhat_debiased, lfsr, What, lfsrW, I_avg, Ihat_avg], index=columns)
    series_to_export.index = columns

    with open('sim_results_PAQu' + version + '.csv','a') as fd:
        writer = csv.DictWriter(fd, fieldnames=columns)
        writer.writerow(series_to_export.to_dict())

if __name__ == '__main__':

    print('Initializing simulations...')
    version = '13'

    # Simulation parameters
    seed_of_seeds = 12
    number_of_seeds = 25
    # seeds = [20, 1, 2001, 21, 6, 2020, 9, 1998, 11, 37] # simulation seeds
    seeds = np.random.RandomState(seed_of_seeds).choice(range(0, 100), number_of_seeds, replace=False).tolist() # simulation seeds
    num_cores = multiprocessing.cpu_count() - 1
    ns = [100, 200, 500] # sample sizes
    easy = False
    if easy: # number of isoforms, number of peptides, number of active isoforms
        qractives = [[1, 2, 1]]
    else:
        qractives = [[5, 10, 1], [5, 10, 2], [5, 10, 3]]
    D_sizes = [1/3, 2/3, 1] # effect size of active isoforms
    D_priors = ['Gaussian', 'Spike-and-Slab']
    gibbs_iters = 3000 # number of iterations in the Gibbs sampler
    burn_in = 2000 # burn-in period
    active_transcripts = True
    active_intercept = True

    # results storer
    columns=['seed', 'n', 'q', 'r', 'D_active', 'D_size', 'D_prior', 'time', 'corrI', 'pihat', 'Dhat', 'Dhat_debiased', 'LFSRD', 'What', 'LFSRW', 'I_avg', 'Ihat_avg']
    results = pd.DataFrame(columns=columns)

    with open('sim_results_PAQu' + version + '.csv','w') as fd:
        writer = csv.writer(fd)
        writer.writerow(columns)

    # Run simulations
    print('Running simulations...')
    Parallel(n_jobs=num_cores)(delayed(fx)(iso) for iso in itertools.product(seeds, ns, qractives, D_sizes, D_priors))