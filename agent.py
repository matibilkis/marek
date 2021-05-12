import basics
import numpy as np
import random

class Agent(basics.Basics):
    def __init__(self, amplitude=0.4, layers=2,
                n_actions=10, number_phases=2,
                bound_displacements=1, searching_method="ep-greedy",
                ep_method="ep-greedy", ep=0.01, min_ep=0.01, tau_ep=500, learning_rate=0, channel={}):

        super().__init__(amplitude=amplitude,layers=layers,
                        n_actions=n_actions, number_phases=number_phases ,
                        bound_displacements=bound_displacements)

        self.searching_method = searching_method
        self.experiments_did=0
        self.ep_method=ep_method

        self.ep=ep
        self.min_ep=min_ep
        self.tau_ep=tau_ep

        self.channel = channel
        self.learning_rate = learning_rate

        self.define_actions()
        self.create_tables()
        self.reset()


    def P(self,a,b,et, outcome):
        if self.channel != {}:
            assert self.channel["class"] == "compound_lossy"
            prob_channel, epsilon = self.channel["params"]
            p0 = 0
            for p_chann, par in zip([prob_channel, 1-prob_channel], [epsilon, 1]):
                p0 += np.exp(-abs((et*a*par)+b)**2)*p_chann
        else:
            p0 = np.exp(-abs((et*a)+b)**2)

        if outcome ==0:
            return p0
        else:
            return 1-p0


    def create_tables(self):

        self.q_table = []
        self.n_table = []

        for layer in range(self.layers):
            indexes = [2]*layer #tree of outcomes possibly obtained so far
            for j in [len(self.actions)]*(layer+1):
                indexes.append(j) #tree of displacements so far + action
            q_table_layer = np.zeros(tuple( indexes))
            n_table_layer = np.zeros(tuple(indexes))

            self.q_table.append(q_table_layer)
            self.n_table.append(n_table_layer)

        self.q_table = np.array(self.q_table)
        self.n_table = np.array(self.n_table)
        #Note: the (2, ) is to append the all possibles n_L (not considered in the q_table)
        self.guess_q_table = np.zeros((2,)+ self.q_table[-1].shape + (self.number_phases,))
        self.guess_n_table = np.zeros( (2,)+ self.q_table[-1].shape + (self.number_phases,))
        return


    def reset(self):
        """Reset agent state to dummy initial state (no partial observations yet)."""
        self.layer=0
        self.actions_index_did = []
        self.actions_value_did=[]
        self.outcomes_observed = []
        return

    def gather_outcome(self,outcome):
        """Put observed outcomes of photodetectors into memory"""
        self.outcomes_observed.append(outcome)
        return

    def give_action_value(self, label, guess=False):
        if not ((isinstance(label,list) or isinstance(label,np.ndarray))):
            if not guess:
                return self.actions[label], label
            else:
                return label, label
        else:
            if len(label)==1:
                label=label[0]
            else:
                label=random.choice(label)
            if not guess:
                action = self.actions[label]
            else:
                action = label
            return action ,label

            return self.actions[label],label



    def select_action(self):
        assert self.searching_method=="ep-greedy"

        if self.ep_method=="exp-decay": #set to whatever otherwise
            self.ep = np.exp(-self.experiments_did/self.tau_ep)
            if self.ep<self.min_ep:
                self.ep=self.min_ep
        r = random.random()
        if (r< self.ep) or (self.ep==1):
            action_index = random.choice(self.action_indexes)
            action = self.actions[action_index]
            self.actions_index_did.append(action_index)
            self.actions_value_did.append(action)
            return action_index, action
        else:
            actual_q = self.q_table[self.layer][ tuple(self.outcomes_observed) ][ tuple(self.actions_index_did)  ]
            greedies = np.where( self.q_table[self.layer][ tuple(self.outcomes_observed) ][ tuple(self.actions_index_did)  ] == np.max(actual_q))[0]
            action, action_index = self.give_action_value(greedies)
            self.actions_index_did.append(action_index)
            self.actions_value_did.append(action)
        return action_index, action



    def q_learn(self, reward):
        """
        Update of the Q-table.
            Note that we also update the parameters of the TS searching both for the guessing stage and also in the case of TS sampling for displacements.

         """

        self.experiments_did +=1

        for layer in range(self.layers+1):
            if layer == self.layers:
                self.guess_n_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] +=1

                learning_rate = 1/self.guess_n_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess]

                times_guessed_here = self.guess_n_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess]
                self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] += (reward - self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] )/times_guessed_here

            else:
                self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])]+=1
                learning_rate = 1/self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])]

                if layer == self.layers-1:
                    target = np.max(self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)]) #choose between 2
                else:
                    target = np.max(self.q_table[layer+1][tuple(self.outcomes_observed[:(layer+1)])][tuple(self.actions_index_did[:(layer+1)])])

                self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += learning_rate*(target- self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])

        return

    def give_guess(self):
        # if (self.method_guess == "ep-greedy")|(self.experiments_did<self.min_actions):
        assert self.searching_method == "ep-greedy"
        if (random.random() < self.ep) or (self.ep==1):
            self.guess = random.choice([0,1])
            return self.guess
        else:
            maxims = np.where(self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)] == np.max(self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)]))[0]
            _, self.guess = self.give_action_value(maxims, guess=True)
            return self.guess


    def greedy_Q_prob(self):
        p=0
        if self.layers == 1:
            l = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
            b, l = self.give_action_value(l)
            for n1 in [0,1]:
                ph = self.possible_phases[np.argmax(self.guess_q_table[n1,l,:])]
                #p+=(1-self.pflip)*self.P(ph*self.amplitude, b ,1, n1) +  self.pflip*self.P(-ph*self.amplitude, b ,1, n1)
                p+=self.P(ph*self.amplitude, b ,1, n1)
            return p/self.number_phases

        elif self.layers == 2:
            l0 = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
            b0, l0 = self.give_action_value(l0)

            l10 = np.where(self.q_table[1][0][l0,:] == np.max(self.q_table[1][0][l0,:]))[0]
            b10, l10 = self.give_action_value(l10)

            l11 = np.where(self.q_table[1][1][l0,:] == np.max(self.q_table[1][1][l0,:]))[0]
            b11, l11 = self.give_action_value(l11)

            for n1,n2 in zip([0,0,1,1],[0,1,0,1]):
                if n1==0:
                    beta2, label2 = b10, l10
                else:
                    beta2, label2 = b11, l11
                ph = np.argmax(self.guess_q_table[n1,n2,l0,label2,:])
                ph = self.possible_phases[ph]

                #p+=(1-self.pflip)*self.P(ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(ph*self.amplitude, beta2 ,1/np.sqrt(2), n2) + (self.pflip*self.P(-ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(-ph*self.amplitude, beta2 ,1/np.sqrt(2), n2))
                # if self.channel != {}:
                #     p+=modified_P(ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*modified_P(ph*self.amplitude, beta2 ,1/np.sqrt(2), n2)
                # else:
                p+=self.P(ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(ph*self.amplitude, beta2 ,1/np.sqrt(2), n2)
            return p/self.number_phases
