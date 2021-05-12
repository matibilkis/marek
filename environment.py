import numpy as np
from tqdm import tqdm
import basics
import os
import shutil
import random

class Environment(basics.Basics):
    def __init__(self, amplitude=.4, layers=2, number_phases=2, channel={}):
        super().__init__(amplitude=amplitude,layers=layers, number_phases=number_phases)

        self.reference_amplitude = amplitude
        self.layers = layers
        if channel != {}:
            self.channel={}
            self.channel["class"] = channel["class"] #lossy, phase_flip...
            self.channel["params"] = channel["params"] #list of params.. [at] if lossy
        self.pick_phase()

    def reset(self):
        self.amplitude = self.reference_amplitude
        return

    def pick_phase(self):
        """Pick a random phase (equal priors) to send to Bob """
        self.phase = random.choices(self.possible_phases)[0]
        self.label_phase = np.where(self.possible_phases == self.phase)[0][0]
        return

    def give_outcome(self, beta,layer):
        effective_attenuation = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])#Warning one!
        probs = [self.P(self.phase*self.amplitude, beta, effective_attenuation, n) for n in [0,1]]
        return random.choices([0,1],weights=probs)[0]

    def give_reward(self, guess):
        if guess == self.label_phase:
            return 1
        else:
            return 0

    def act_channel(self):
        if hasattr(self, "channel"):
            if self.channel["class"] == "compound_lossy":
                prob, epsilon = self.channel["params"]
                if np.random.random() <= prob:
                    self.amplitude *=np.sqrt(epsilon)
        return


    def lambda_q(self,q):
        """Auxiliary method to compute pretty good measurement bound (helstrom in this case, see Holevo book)"""
        number_states = self.number_phases
        nsig = self.amplitude**2 #in case you change...
        c=0
        for m in range(1,number_states+1):
            c+= np.exp(((1-q)*(2*np.pi*(1j)*m)/number_states) + nsig*np.exp(2*np.pi*(1j)*m/number_states))
        return c*np.exp(-nsig)


    def helstrom(self):
        """
        Returns helstrom probability sucess
        Eq (9) M-ary-state phase-shift-keying discrimination below the homodyne limit
        F. E. Becerra,1,* J. Fan,1 G. Baumgartner,2 S. V. Polyakov,1 J. Goldhar,3 J. T. Kosloski,4 and A. Migdall1
        """
        nsig=self.amplitude**2
        number_states=self.number_phases

        prob = 0
        for q in range(1,number_states+1):
            prob += np.sqrt(self.lambda_q(q))
        prob = 1 - (prob/number_states)**2

        return np.real(1-prob)




#####
