%load_ext autoreload
%autoreload 2

import numpy as np
from dynamic import SuccessProbability, RewardModel
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize


class InterpolationModel:
    def __init__(self, priors, labels):
        self.model = interp1d(priors, labels)
    def __call__(self, unseen_point):
        if unseen_point < .5:
            return self.model(1-unseen_point)
        else:
            return self.model(unseen_point)

class RewardModel():
    def __init__(self):
        pass
    def R_function(self,postirior_plus):
        reward = np.max([postirior_plus, 1-postirior_plus])
        return reward
    def __call__(self,postiriors):
        return self.R_function(postiriors)



class SuccessProbability():
    def __init__(self, n_phases=2,amplitude=0.4):
        self.n_phases = n_phases
        self.amplitude = amplitude
        self.possible_phases = np.array([-1.,1.])

    def NProb(self,outcome, amp,beta):
        p0=np.exp(-abs(complex(amp)+complex(beta))**2)
        if outcome == 0.:
            return p0
        else:
            return 1-p0

    def outcome_probability(self,outcome,prior_plus,beta,amp):
        prs = [prior_plus, 1-prior_plus]
        p=0
        for phase,pr in zip([1,-1], prs):
            p+= pr*self.NProb(outcome,phase*amp,beta)
        return p

    def postirior_probability(self,outcome, prior_plus, beta, amp):
        prob_ot = self.outcome_probability(outcome, prior_plus, beta, amp)
        return prior_plus*self.NProb(outcome, amp,beta)/prob_ot

    def J_intermmediate(self, prior_plus, beta, amplitude, next_J_interpolation):
        objective_function = 0
        for outcome in [0, 1]:
            postirior_plus = self.postirior_probability(outcome, prior_plus, beta, amplitude)
            objective_function += next_J_interpolation(postirior_plus)*self.outcome_probability(outcome, prior_plus, beta, amplitude)
        return -objective_function


amplitude = 0.4
number_photodetectors = 2
Npriors = 25
priors = np.linspace(.5,1,Npriors)

objective_functions = np.zeros((number_photodetectors+1,len(priors)))


### last layer ###
modelito = RewardModel()
for indp, pr in enumerate(priors):
    objective_functions[number_photodetectors, indp] = modelito(pr)

for layer in range(number_photodetectors)[::-1]:
    suc = SuccessProbability(amplitude=amplitude/np.sqrt(number_photodetectors))
    mod = InterpolationModel(priors,objective_functions[layer+1])
    for indp, pr in enumerate(priors):
        def suc1(beta):
            return suc.J_intermmediate(pr, beta[0], 0.4, mod)
        objective_functions[layer,indp] = -optimize.dual_annealing(suc1, [(-1,1)], maxiter=200,no_local_search=True ).fun
