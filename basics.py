import numpy as np
import misc
from math import erf

class Basics():
    def __init__(self, amplitude = 0.4, layers=2,
        number_phases=2, bound_displacements=1, n_actions=10):

        self.layers = layers
        self.number_phases = number_phases
        self.at = self.make_attenuations(self.layers) #Bob uses this if she knows the model
        self.amplitude = amplitude
        self.possible_phases=np.array([-1,1])
        self.bound_displacements=bound_displacements
        self.n_actions=n_actions

    def make_attenuations(self,layers):
        if layers == 1:
            return [0]
        else:
            ats=[0]
            for i in range(layers-1):
                ats.append(np.arctan(1/np.cos(ats[i])))
            return ats[::-1]

    def define_actions(self):
        self.actions = np.linspace(-self.bound_displacements, self.bound_displacements, self.n_actions)
        self.action_indexes = np.arange(0,len(self.actions))
        return

    def P(self,a,b,et,n):
        p0=np.exp(-abs((et*a)+b)**2)
        if n ==0:
            return p0
        else:
            return 1-p0

    def err_kennedy(self,beta):
        return (1 + np.exp(- (-beta + self.amplitude)**2)  - np.exp(- (beta + self.amplitude)**2)  )/2

    def homodyne(self):
        return (1+erf(self.amplitude))/2

    def heterodyne(self):
        return (1+(1-np.exp(-2*self.amplitude**2))/np.sqrt(np.pi))/2

    def probability_error(self, betas):
        outcomes = misc.outcomes_universe(self.layers)
        B=np.zeros((self.layers,2**(self.layers-1)),dtype=complex)
        for k in range(self.layers):
            B[k,:] = np.repeat(betas[2**k -1:2**(k+1)-1],2**(self.layers-k-1),axis=0).flatten()
        p,k=0,0
        for i in range(outcomes.shape[0]):
            ot = outcomes[i]
            if self.layers==1:
                ot = [ot]
            if i%2==0:
                bb = B[:,k]
                k+=1
            term =0
            compare = []
            for phase in misc.croots(self.number_phases):
                term = 1
                for j in range(self.layers):
                    r =  np.prod( np.sin(self.at[:j]))
                    t = np.cos(self.at[j])

                    term = term*self.P(phase*self.amplitude,bb[j],r*t, ot[j])
                compare.append(term)
            p+=np.max(compare)
        return 1- (p/self.number_phases)
