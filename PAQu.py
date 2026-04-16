#
# PAQu Class
#

import time
import numpy as np
import scipy.stats as stats
from typing import Union

from tqdm import tqdm

class PAQu():

    def __init__(self, A:Union[np.array, None], M:np.array, P:np.array, T:Union[np.array, int], Xi:Union[np.array, None]=None, Xp:Union[np.array, None]=None):
        # Known data
        self.P = P
        self.M = M

        # Dimensions
        self.n = P.shape[0]
        self.r = P.shape[1]

        # Transcripts
        if isinstance(T, np.ndarray):
            self.T = T
            self.fit_transcript = True
            self.q = T.shape[1]
        else:
            self.q = T
            self.T = np.zeros((self.n, self.q))
            self.fit_transcript = False

        # Dummy variable
        if isinstance(A, np.ndarray):
            self.A = A.ravel()
            self.fit_dummy = True
        else:
            self.A = np.zeros((self.n))
            self.fit_dummy = False

        # External covariates
        if isinstance(Xi, np.ndarray):
            self.Xi = Xi
            if self.Xi.ndim == 1:
                self.Xi = self.Xi.reshape(self.n, 1)
            self.fit_covariates_i = True
        else:
            self.Xi = np.zeros((self.n, 1))
            self.fit_covariates_i = False
        if isinstance(Xp, np.ndarray):
            self.Xp = Xp
            self.fit_covariates_p = True
        else:
            self.Xp = np.zeros((self.n, 1))
            self.fit_covariates_p = False
        self.pXi = self.Xi.shape[1]
        self.pXp = self.Xp.shape[1]

        # Intercept
        self.ones = np.ones((self.n))

        # Initialize parameters
        if self.q <= self.r:
            svd = np.linalg.svd(self.P)
            self.I = svd[0][:, :self.q] @ np.diag(svd[1][:self.q])# left singular vectors
            self.Z = svd[2][:self.q, :] * self.M # masked right singular vectors
        else:
            self.I = self.T
            self.Z = np.ones((self.q, self.r)) * self.M
        self.I0 = np.zeros((self.q))
        self.W = np.zeros((self.q))
        self.D = np.zeros((self.q))
        self.Bi = np.zeros((self.pXi, self.q))
        self.Bp = np.zeros((self.pXp, self.r)) # FIX
        for j in range(self.q):
            X_design = np.column_stack([np.ones(self.n), self.A, self.T[:,j], self.Xi])
            ols = np.linalg.lstsq(X_design, self.I[:,j], rcond=None)
            self.I0[j] = ols[0][0]
            self.D[j] = ols[0][1]
            self.W[j] = ols[0][2]
            self.Bi[:,j] = ols[0][3:]
        
        # Hyperparameters
        self.sigmaI = np.ones((self.q))
        self.sigmaP = np.ones((self.r))
        self.tauI0 = np.ones((self.q))
        self.tauD = np.ones((self.q))
        self.tauI = np.ones((self.q))
        self.tauW = np.ones((self.q))
        self.pi = np.ones((self.q)) * 0.2
        self.theta = np.ones((self.q)) / 2
        self.tauZ = np.ones((self.q))
        self.tauBi = np.ones((self.pXi))
        self.tauBp = np.ones((self.pXp))

        # Default hyperparameters
        self.alphaThetaj = 1
        self.betaThetaj = 1
        self.scaleDj = 1
        self.shapeDj = 1
        self.meanDj = 0
        self.shapeI0 = 1
        self.scaleI0 = 1
        self.meanI0j = 0
        self.shapeBil = 1
        self.scaleBil = 1
        self.meanBil = 0
        self.scaleWj = 1
        self.shapeWj = 1
        self.meanWj = 0
        self.shapeSigmaIj = 1
        self.scaleSigmaIj = 1
        self.shapeZ = 1
        self.scaleZ = 1
        self.meanZj = 1
        self.shapeBpl = 1
        self.scaleBpl = 1
        self.meanBpl = 0
        self.shapeSigmaP = 1
        self.scaleSigmaP = 1
        self.shapeIj = 1
        self.scaleIj = 1

        # Storers
        self.I0_storer = None
        self.I_storer = None
        self.W_storer = None
        self.D_storer = None
        self.Z_storer = None
        self.pi_storer = None
        self.Bi_storer = None
        self.Bp_storer = None
        self.sigmaI_storer = None
        self.sigmaP_storer = None
        self.tauD_storer = None
        self.tauI_storer = None
        self.tauBi_storer = None
        self.tauBp_storer = None
        self.tauZ_storer = None
        self.tauW_storer = None
        self.theta_storer = None

        self.fit_time = 0
        self.fit_intercept = False
        self.fit_tauI = False
        self.prior_D = 'Gaussian'
        self.verbose = False

        # Z prior specifics
        self.slack = 0.1
        self.lower = self.P.mean().min() / self.P.mean().max()
        self.upper = self.P.mean().max() / self.P.mean().min()

    def _gibbs_step(self):
        """
        Perform one iteration of the Gibbs sampler.
        
        Parameters:
        -----------
        None
        
        Returns:
        --------
        None
        """

        def sampletheta(pij):
            return stats.beta.rvs(self.alphaThetaj + pij, self.betaThetaj + 1 - pij)
        
        # def sampletheta(pi):
        #     return stats.beta.rvs(self.alphaThetaj + np.sum(pi), self.betaThetaj + self.q - np.sum(pi))
                
        def sampletauD(Dj, pij, sigmaIj):
            if self.fit_dummy is False:
                return 1
            # return stats.invgamma.rvs(self.shapeDj + pij/2, scale=self.scaleDj + (Dj**2)/(2 * sigmaIj))
            return stats.invgamma.rvs(self.shapeDj + pij/2, scale=self.scaleDj + (Dj**2)/2)

        def sampletauW(sigmaIj, Wj):
            if self.fit_transcript is False:
                return 1
            # return stats.invgamma.rvs(self.shapeWj + 1/2, scale=self.scaleWj + (Wj**2)/(2 * sigmaIj))
            return stats.invgamma.rvs(self.shapeWj + 1/2, scale=self.scaleWj + (Wj**2)/2)

        def sampletauBi(Bil):
            if self.fit_covariates_i is False:
                return 1
            # return stats.invgamma.rvs(self.shapeWj + 1/2, scale=self.scaleWj + (Wj**2)/(2 * sigmaIj))
            return stats.invgamma.rvs(self.shapeBil + self.pXi/2, scale=self.scaleBil + np.sum(Bil**2)/2)

        def sampletauI0(I0j, sigmaIj):
            if self.fit_intercept is False:
                return 1
            # return stats.invgamma.rvs(self.shapeI0 + 1/2, scale=self.scaleI0 + (I0j**2)/(2 * sigmaIj))
            return stats.invgamma.rvs(self.shapeI0 + 1/2, scale=self.scaleI0 + (I0j**2)/2)
        
        def samplesigmaI(A, Bij, Dj, Ij, I0j, Tj, Wj, Xi):
            return stats.invgamma.rvs(self.shapeSigmaIj + self.n/2, scale=self.scaleSigmaIj + (1/2) * np.linalg.norm(Ij - self.ones * I0j - A * Dj - Tj * Wj - (Xi @ Bij).ravel(), ord=2)**2)
            # return stats.invgamma.rvs(self.shapeSigmaIj + self.n/2, scale=self.scaleSigmaIj + (1/2) * np.linalg.norm(Ij - A * Dj, ord=2)**2)

        def sampleW(A, Bij, Dj, Ij, I0j, sigmaIj, tauWj, Tj, Xi):
            if self.fit_transcript is False:
                return 0
            # var = sigmaIj / ((1/tauWj) + (1/sigmaIj) * np.linalg.norm(Tj, ord=2)**2)
            var = 1 / ((1/tauWj) + (1/sigmaIj) * np.linalg.norm(Tj, ord=2)**2)
            mean = var * ((self.meanWj/tauWj) + (1/sigmaIj) * np.sum(Tj * (Ij - self.ones * I0j - A * Dj - (Xi @ Bij).ravel())))
            return stats.norm.rvs(mean, np.sqrt(var))

        def sampleD(A, Bij, Ij, I0j, pij, sigmaIj, tauDj, Tj, Wj, Xi):
            if self.fit_dummy is False:
                return 0
            if pij == 0:
                return 0
            # var = sigmaIj / ((1/tauDj) + (1/sigmaIj) * np.linalg.norm(A, ord=2)**2)
            var = 1 / ((1/tauDj) + (1/sigmaIj) * np.linalg.norm(A, ord=2)**2)
            mean = var * ((self.meanDj/tauDj) + (1/sigmaIj) * np.sum(A * (Ij - self.ones * I0j - Tj * Wj - (Xi @ Bij).ravel())))
            # mean = np.sum(A * Ij) / ((1/tauDj) + (1/sigmaIj) * np.linalg.norm(A, ord=2)**2)
            return stats.norm.rvs(mean, np.sqrt(var))

        def sampleBi(A, Bij, Bilj, Dj, Ij, I0j, sigmaIj, tauBil, Tj, Wj, Xi, Xil):
            if self.fit_covariates_i is False:
                return 0
            var = 1 / ((1/tauBil) + (1/sigmaIj) * np.linalg.norm(Xil, ord=2)**2)
            mean = var * ((self.meanBil/tauBil) + (1/sigmaIj) * np.sum(Xil * (Ij - self.ones * I0j - A * Dj - Tj * Wj - (Xi @ Bij).ravel() + Xil * Bilj)))
            return stats.norm.rvs(mean, np.sqrt(var))
            
        def sampleI0(A, Bij, Dj, Ij, sigmaIj, tauI0j, Tj, Wj, Xi):
            if self.fit_intercept == False:
                return 0
            #var = sigmaIj / ((1/tauI0j) + (1/sigmaIj) * np.linalg.norm(Ij, ord=2)**2)
            var = 1 / ((1/tauI0j) + (1/sigmaIj) * self.n)
            mean = var * ((self.meanI0j/tauI0j) + (1/sigmaIj) * np.sum(Ij - A * Dj - Tj * Wj - (Xi @ Bij).ravel()))
            return stats.norm.rvs(mean, np.sqrt(var))

        def samplepi(A, Bij, Ij, I0j, sigmaIj, tauDj, thetaj, Tj, Wj, Xi):
            # I divide the denominator in 3 blocks to avoid numerical instability
            # block_exp = np.exp(np.sum((Ij - self.ones * I0j - Tj * Wj) * A)**2 / (2 * sigmaIj**2 * (1/tauDj + np.linalg.norm(A, ord=2) ** 2)/sigmaIj))
            # block_sqrt = np.sqrt(1 / (1/tauDj + (np.linalg.norm(A, ord=2) ** 2) / sigmaIj))
            # p = 1 - (1 - thetaj) / ((1/np.sqrt(tauDj)) * block_exp * block_sqrt * thetaj + 1 - thetaj)

            # log_odds = - 0.5 * np.log(tauDj)
            # log_odds+= np.sum((Ij - self.ones * I0j - Tj * Wj) * A)**2 / (2 * sigmaIj**2 * (1/tauDj + np.linalg.norm(A, ord=2) ** 2)/sigmaIj)
            # log_odds+= 0.5 * np.log(1 / (1/tauDj + (np.linalg.norm(A, ord=2) ** 2) / sigmaIj))
            # log_odds+= np.log(thetaj)
            # p = 1 - (1-thetaj) / (1 - thetaj + np.exp(log_odds))

            # log_odds-= np.log(1 - thetaj)
            # p = 1 / (1 + np.exp(-log_odds))

            # block_exp = np.exp(np.sum((Ij - Tj * Wj) * A)**2 / (2 * sigmaIj * (1/tauDj + np.linalg.norm(A, ord=2) ** 2)))
            # block_sqrt = np.sqrt(sigmaIj / (1/tauDj + np.linalg.norm(A, ord=2) ** 2))
            # p = (1 - thetaj) / ((1/np.sqrt(sigmaIj * tauDj)) * block_exp * block_sqrt * thetaj + 1 - thetaj)

            # l0 = np.log(1 - thetaj)
            # sum_xy = np.sum((Ij - self.ones * I0j - Tj * Wj) * A)**2
            # cond_var = (1/tauDj) + np.sum(A**2)
            # l1 = np.log(thetaj) - 0.5 * np.log(tauDj * sigmaIj) + sum_xy / (2 * sigmaIj * cond_var) + 0.5 * np.log(sigmaIj / cond_var)
            # p = np.exp(l1) / (np.exp(l0) + np.exp(l1))

            if self.prior_D == 'Gaussian':
                return 1

            # As in GSFA
            log_odds = 0.5 * np.log(1 / (tauDj * (1/tauDj + np.sum(A**2))))
            log_odds+= np.sum((Ij - self.ones * I0j - Tj * Wj - (Xi @ Bij).ravel()) * A)**2 / (2 * (1/tauDj + np.sum(A**2)))
            log_odds+= np.log(thetaj) - np.log(1 - thetaj)
            if log_odds > 30:
                return 1
            p = 1 / (1 + np.exp(-log_odds))
            return stats.bernoulli.rvs(p)
        
        def sampletauZ(Mj, Zj):
            return stats.invgamma.rvs(self.shapeZ + np.sum(Mj)/2, scale=self.scaleZ + (1/2) * np.linalg.norm(Zj, ord=2)**2)
        
        def sampletauBp(Bpk):
            if self.fit_covariates_p is False:
                return 1
            return stats.invgamma.rvs(self.shapeBpl + self.pXp/2, scale=self.scaleBpl + np.sum(Bpk**2)/2)

        def sampleZ(Bpk, I, Ij, Mjk, Pk, sigmaPk, tauZj, Xp, Zjk, Zk, lower=0, upper=1):
            if Mjk == 0:
                return 0
            var = 1 / ((1/tauZj) + (1/sigmaPk) * np.linalg.norm(Ij, ord=2)**2)
            mean = var * ((self.meanZj/tauZj) + (1/sigmaPk) * np.sum((Pk - np.dot(I, Zk) + Ij * Zjk - (Xp @ Bpk).ravel()) * Ij))
            # return stats.norm.rvs(mean, np.sqrt(var))
            a, b = (lower - mean) / np.sqrt(var), (upper - mean) / np.sqrt(var)
            return stats.truncnorm.rvs(a, b, loc=mean, scale=np.sqrt(var))
            # return np.abs(stats.norm.rvs(mean, np.sqrt(var)))
        
        # def sampleZ_MH(Bpk, I, Ij, Mjk, Pk, sigmaPk, tauZj, Xp, Zjk, Zk):
        #     if Mjk == 0:
        #         return 0
        #     # Implement Metropolis-Hastings for Zjk, gamma prior, normal likelihood, normal proposal
        #     # Proposal
        #     Zjk_prop = np.abs(stats.norm.rvs(Zjk, 0.1))
        #     # Likelihood
        #     l1 = - 0.5 * np.sum(Pk - np.dot(I, Zk) + Ij * Zjk - Xp @ Bpk)**2 / sigmaPk
        #     l2 = - 0.5 * np.sum(Pk - np.dot(I, Zk) + Ij * Zjk_prop - Xp @ Bpk)**2 / sigmaPk
        #     # Prior
        #     l1+= np.log(stats.gamma.pdf(Zjk, self.shapeZ, scale=self.scaleZ))
        #     l2+= np.log(stats.gamma.pdf(Zjk_prop, self.shapeZ, scale=self.scaleZ))
        #     # Acceptance
        #     alpha = np.exp(l2 - l1)
        #     if alpha > 1:
        #         Zjk = Zjk_prop
        #     else:
        #         if np.random.uniform() < alpha:
        #             Zjk = Zjk_prop
        #     return Zjk
        
        def sampleBp(Bpk, Bplk, I, Pk, tauBpl, sigmaPk, Xp, Xpl, Zk):
            if self.fit_covariates_p is False:
                return 0
            var = 1 / ((1/tauBpl) + (1/sigmaPk) * np.linalg.norm(Xpl, ord=2)**2)
            mean = var * ((self.meanBpl/tauBpl) + (1/sigmaPk) * np.sum((Pk - np.dot(I, Zk) - np.dot(Xp, Bpk) + Xpl * Bplk) * Xpl))
            return stats.norm.rvs(mean, np.sqrt(var))

        def samplesigmaP(Bpk, I, Pk, Xp, Zk):
            IZk = I @ Zk
            XpBpk = Xp @ Bpk
            return stats.invgamma.rvs(self.shapeSigmaP + self.n/2, scale=self.scaleSigmaP + (1/2) * np.linalg.norm(Pk - IZk - XpBpk, ord=2)**2)
        
        def sampletauI(Ij):
            if self.fit_tauI is False:
                return 1
            return stats.invgamma.rvs(self.shapeIj + self.n/2, scale=self.scaleIj + np.linalg.norm(Ij, ord=2)**2/2)
        
        def sampleI(Ai, Bij, Bp, Dj, Ii, Iij, I0j, Pi, sigmaP, Tij, tauIj, Wj, Xii, Xpi, Z, Zj):
            var = 1 / ((1/tauIj) + np.sum(Zj**2 / sigmaP))
            mean = var * (((I0j + Tij * Wj + Ai * Dj + Xii @ Bij)/tauIj) + np.sum((Pi - Ii @ Z + Iij * Zj - (Xpi @ Bp).ravel()) * Zj / sigmaP))
            return stats.norm.rvs(mean, np.sqrt(var))

        # Independent step -- I = I0 + TW + AD + XB + E
        for j in range(self.q):
            self.W[j] = sampleW(self.A, self.Bi[:,j], self.D[j], self.I[:, j], self.I0[j], self.sigmaI[j], self.tauW[j], self.T[:, j], self.Xi)
            self.tauW[j] = sampletauW(self.sigmaI[j], self.W[j])
            self.pi[j] = samplepi(self.A, self.Bi[:,j], self.I[:, j], self.I0[j], self.sigmaI[j], self.tauD[j], self.theta[j], self.T[:, j], self.W[j], self.Xi)
            self.D[j] = sampleD(self.A, self.Bi[:,j], self.I[:, j], self.I0[j], self.pi[j], self.sigmaI[j], self.tauD[j], self.T[:, j], self.W[j], self.Xi)
            self.tauD[j] = sampletauD(self.D[j], self.pi[j], self.sigmaI[j])
            self.I0[j] = sampleI0(self.A, self.Bi[:,j], self.D[j], self.I[:, j], self.sigmaI[j], self.tauI0[j], self.T[:, j], self.W[j], self.Xi)
            self.tauI0[j] = sampletauI0(self.I0[j], self.sigmaI[j])
            for l in range(self.pXi):
                self.Bi[l, j] = sampleBi(self.A, self.Bi[:,j], self.Bi[l,j], self.D[j], self.I[:,j], self.I0[j], self.sigmaI[j], self.tauBi[l], self.T[:,j], self.W[j], self.Xi, self.Xi[:,l])
            self.sigmaI[j] = samplesigmaI(self.A, self.Bi[:,j], self.D[j], self.I[:, j], self.I0[j], self.T[:, j], self.W[j], self.Xi)

        for l in range(self.pXi):
            self.tauBi[l] = sampletauBi(self.Bi[l,:])

        # st = sampletheta(self.pi)
        for j in range(self.q):
            st = sampletheta(self.pi[j])
            self.theta[j] = st

        # Dependent step (Z) -- P = IZ + XB + E
        for k in range(self.r):
            for j in range(self.q):
                self.Z[j,k] = sampleZ(self.Bp[:,k], self.I, self.I[:, j], self.M[j, k], self.P[:, k], self.sigmaP[k], self.tauZ[j], self.Xp, self.Z[j, k], self.Z[:, k], self.lower - self.slack, self.upper + self.slack)
            for l in range(self.pXp):
                self.Bp[l,k] = sampleBp(self.Bp[:,k], self.Bp[l,k], self.I, self.P[:,k], self.tauBp[l], self.sigmaP[k], self.Xp, self.Xp[:,l], self.Z[:,k])

        for j in range(self.q):
            self.tauZ[j] = sampletauZ(self.M[j, :], self.Z[j, :])
        for l in range(self.pXp):
            self.tauBp[l] = sampletauBp(self.Bp[l, :])

        # Dependent step (I) -- P = IZ + XB + E
        for j in range(self.q):
            for i in range(self.n):
                self.I[i, j] = sampleI(self.A[i], self.Bi[:,j], self.Bp, self.D[j], self.I[i, :], self.I[i, j], self.I0[j], self.P[i, :], self.sigmaP, self.T[i, j], self.tauI[j], self.W[j], self.Xi[i,:], self.Xp[i,:], self.Z, self.Z[j, :])
            self.tauI[j] = sampletauI(self.I[:, j])

        for k in range(self.r):
            self.sigmaP[k] = samplesigmaP(self.Bp[:,k], self.I, self.P[:, k], self.Xp, self.Z[:, k])

    def fit(self, n_iter:int, fit_intercept=False, fit_tauI=False, prior_D:str='Spike-and-Slab', verbose=False):
        """
        Fit the model using Gibbs sampling.

        Parameters:
        -----------
        n_iter: int
            Number of iterations to run the Gibbs sampler.
        
        Returns:
        --------
        None
        """

        self.fit_intercept = fit_intercept
        self.fit_tauI = fit_tauI
        self.prior_D = prior_D
        self.verbose = verbose

        self.I_storer = np.zeros((n_iter, self.n, self.q))
        self.W_storer = np.zeros((n_iter, self.q))
        self.D_storer = np.zeros((n_iter, self.q))
        self.Z_storer = np.zeros((n_iter, self.q, self.r))
        self.pi_storer = np.zeros((n_iter, self.q))
        self.I0_storer = np.zeros((n_iter, self.q))
        self.Bi_storer = np.zeros((n_iter, self.pXi, self.q))
        self.Bp_storer = np.zeros((n_iter, self.pXp, self.r))
        self.sigmaI_storer = np.zeros((n_iter, self.q))
        self.sigmaP_storer = np.zeros((n_iter, self.r))
        self.tauD_storer = np.zeros((n_iter, self.q))
        self.tauBi_storer = np.zeros((n_iter, self.pXi))
        self.tauBp_storer = np.zeros((n_iter, self.pXp))
        self.tauZ_storer = np.zeros((n_iter, self.q))
        self.tauW_storer = np.zeros((n_iter, self.q))
        self.theta_storer = np.zeros((n_iter, self.q))
        self.tauI_storer = np.zeros((n_iter, self.q))

        print('Fitting PAQu')

        start = time.time()
        for it in tqdm(range(n_iter)):
            self._gibbs_step()

            self.I_storer[it] = self.I
            self.W_storer[it] = self.W
            self.D_storer[it] = self.D
            self.Bi_storer[it] = self.Bi
            self.Z_storer[it] = self.Z
            self.pi_storer[it] = self.pi
            self.I0_storer[it] = self.I0
            self.Bp_storer[it] = self.Bp
            self.sigmaI_storer[it] = self.sigmaI
            self.sigmaP_storer[it] = self.sigmaP
            self.tauD_storer[it] = self.tauD
            self.tauBi_storer[it] = self.tauBi
            self.tauBp_storer[it] = self.tauBp
            self.tauZ_storer[it] = self.tauZ
            self.tauW_storer[it] = self.tauW
            self.theta_storer[it] = self.theta
            self.tauI_storer[it] = self.tauI
        end = time.time()
        self.fit_time = end - start
     
    def update_hyperparameters(self, name, value):
        """
        Given the name of a hyperparameter and a new value, update the hyperparameter.

        Parameters:
        -----------
        name: str
            Name of the hyperparameter to update.
        value: float
            New value of the hyperparameter.

        Returns:
        --------
        None
        """
        setattr(self, name, value)

    def LFSR(self, storer, burn_in):
        """
        Compute the LFSR for a given storer.

        Parameters:
        -----------
        storer: str
            Name of the storer to compute the LFSR.
        burn_in: int
            Number of burn-in iterations.
        
        Returns:
        --------
        float
            LFSR for the given storer.
        """
        
        box = getattr(self, storer)[burn_in:,]
        lfsr = np.minimum(np.mean(box >= 0, axis=0), np.mean(box <= 0, axis=0))
        return lfsr