import misc
import numpy as np
import scipy.optimize as sp
from tqdm import tqdm
import basics
import os
import shutil
import random
try:
    import strawberryfields as sf
    from strawberryfields.ops import *
except Exception:
    pass
class Environment(basics.Basics):
    """

    Environment class. The outcomes at each photodetector as well as the displacements can be implemented with the Strawberryfields simulator. Despite this being slower, it is a proof of principle and ideally would be implemented in an optical table controlled by this software. See give_outcome_sf method.

    ** amplitude: mean energy
    ** std: suppose you have a gaussian distribution of energy, std is standard deviation
    ** layers: #photodetectors
    ** resolution
    ** number_phases (Alice's alphabet)
    ** bound_displacements
    ** how: the attenuations are constructed such that the energy of the state that arrives to any photodetector is equal.
    This can be changed in the method how of function make_attenuations_equal_intensity, but in principle this is optimal (To check with DP)

    """

    def __init__(self, amplitude=.4, std=0, layers=2, resolution=0.1, number_phases=2, bound_displacements=1,  how="equal_energy_detected",efficiency=1,pflip=0):
        super().__init__(amplitude=amplitude,layers=layers, resolution=resolution, number_phases=number_phases, bound_displacements=bound_displacements, efficiency =efficiency,pflip=pflip)

        self.amplitude = amplitude
        self.layers = layers
        self.std = std
        self.mean=self.amplitude #self.amplitude may change along different experiments..
        self.pick_phase()


    def lambda_q(self,q):
        """Auxiliary method to compute pretty good measurement bound (helstrom in this case, see Holevo book)"""
        number_states = self.number_phases
        nsig = self.mean**2 #in case you change...
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
        if self.layers ==1:
            return 1-np.min(self.err_kennedy(self.actions[0]))

        nsig=self.mean**2
        number_states=self.number_phases

        prob = 0
        for q in range(1,number_states+1):
            prob += np.sqrt(self.lambda_q(q))
        prob = 1 - (prob/number_states)**2

        return 1-prob

    def pick_phase(self):
        """Pick a random phase (equal priors) to send to Bob """
        self.phase = random.choices(self.possible_phases)[0]
        self.flipped = False

        if random.random() <= self.pflip:
            self.phase = -self.phase
            self.flipped = True

        self.label_phase = np.where(self.possible_phases == self.phase)[0][0]
        if self.std!=0:
            self.amplitude = np.random.normal(self.mean, self.std, 1)[0]
        return

    def give_outcome(self, beta,layer):
        """ Returns outcome according to current layer (needed to compute the current intensity)""" #Actually, if all intensities are equal, you don't need to keep track of the layer here...

        effective_attenuation = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])#Warning one!
        probs = [self.P(self.phase*self.amplitude, beta, effective_attenuation, n) for n in [0,1]]

        return random.choices([0,1],weights=probs)[0]

    def give_outcome_sf(self, beta, layer):
        """ Returns outcome according to current layer (needed to compute the "current" intensity).
            To accomplish this, it is used the Strawberryfields simulator, as a proof of principle that this
            can be easily done with this photonic platform.

            Notice that - if desired - the full experiment could be implemented with strawberryfields, considering a number of self.layers modes, and applying the displacements with the corresponding feed-forward. For the moment, as only the outcomes at each photodetector are needed to learn the correct displacements, we obtained them separately.

        """ #Actually, if all intensities are equal, you don't need to keep track of the layer here...
        effective_attenuation = np.prod(np.sin(self.at[:layer]))*np.cos(self.at[layer])#Warning one!

        eng = sf.Engine("fock", backend_options={"cutoff_dim": 4})
        prog = sf.Program(1)

        with prog.context as q:
            Coherent(a=(effective_attenuation*self.phase*self.amplitude)-beta) | q[0]
            MeasureFock() | q[0]
        results = eng.run(prog)

        outcome = np.array(outcome_from_fock(results).samples)[0]
        if outcome==0:
            return 0
        else:
            return 1

    def give_reward(self, guess):
        """We put label_phase to avoid problems
        with the np.round we applied to complex phases"""
        if (self.flipped):
            self.label_phase = np.where(self.possible_phases == -self.phase)[0][0]

        if guess == self.label_phase:
            return 1
        else:
            return 0
