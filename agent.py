import basics
import numpy as np
from misc import outcomes_universe
import random

class Agent(basics.Basics):
    """Class Agent.


        * layers, resolution, number_phases, bound_displacements are methods of the class Basics

        * guessing_rule: how the Agent decides which phase Alice sent
                    1) "None" ---> he just learn by itself a guessing rule, using TS.

                        The guessing rule if None, is done via Thompson Sampling. You keep a prior distribution on how good is to bet
                        for each phase, given each trajectory of states actions seen. You make your bet by sampling from each of the N phase-priors-distributions you have for the current
                        trajectory of states actions, by going for the highest result.

                    2) "Dolinar" ---> works for 2 phases, see Dolinar paper, or my draft called "dolinar_works"
                    3) "max-likelihood" ---> the Agent knows the model, and makes a bet according to maximum likelihood. Notice that in this option, the agent also constructs his own reward, which is now "a branch" of the probability of success.

        * searching_method:   Agent's policy for picking actions given a state.
                        ep-greedy
                        ucb
                        thompson-sampling

        ** ep_method:
            If ep_method="exp-decay", ep(t) = exp(-t/tau) or .001

        ** ucb_method:
            ucb1, ucb2, ucb3 (changes delta(t)), ucb4: kl-ucb

        ** ep:
            value of epsilon if ep_method != "exp-decay"

        ** soft_ts:
            soften the update rule in thompson sampling

        ** learning_rate:
            learning rate (note that may be fixed to number of times visited).
            if set to 0, then use the number of times visited

        **method_guess: thompson-sampling

        ** algortithm: whether to use UCB-algorithm or not! borrowred from Is Q-learning provably efficient ?
            *if not "UCB-eff", then goes with something (whatever) else
            * strange_factor_ucbeff, *prob_eff (see training.py)

        ** ts_method= update according to the current q-value! (if = update_to_q)
    """



    def __init__(self, layers=2, resolution=0.1, number_phases=2, bound_displacements=1, guessing_rule="None", searching_method="ep-greedy", ep_method="exp-decay", ucb_method="ucb1", ep=0.01, min_ep=0.01, soft_ts = 1, learning_rate=0, efficiency=1, method_guess="undefined", time_tau=1000, pflip=0, algorithm="standard", strange_factor_ucbeff=1, prob_eff=0.01, ts_method="std"):

        super().__init__(layers=layers, resolution=resolution, number_phases=number_phases , bound_displacements=bound_displacements, efficiency=efficiency,pflip=pflip)
        self.method = searching_method
        self.experiments_did=0
        self.guessing_rule = guessing_rule
        self.ep_method=ep_method
        self.ts_method = ts_method
        self.min_actions = -1 #minimum number of actions to do before beggining the method... we didn't use it in the end


        self.ep=ep
        self.ep_saved = ep #i save the data to do exploration first episodes, just in case i save the fist episode (sometimes i don't define ep_saved)
        self.min_ep=min_ep
        self.ucb_method = ucb_method
        self.soft_ts = soft_ts

        if method_guess == "undefined":
            self.method_guess = self.method
        else:
            self.method_guess = method_guess

        if self.method == "ucb":
            if self.ucb_method not in ["ucb1", "ucb2", "ucb3", "ucb4"]:
                print("Ill-defined ucb_method, we'll use ucb1")
                self.ucb_method = "ucb1"
        self.time_tau = time_tau

        self.learning_rate = learning_rate

        self.define_actions()

        self.iota = np.log(2**(self.layers+1)*len(self.actions)/prob_eff) #Regret MDPs, didn't work
        self.c = strange_factor_ucbeff
        self.algorithm=algorithm
        self.create_tables()
        self.reset()



    def create_tables(self):

        """
        Define all tables:
        N-table (table of visits for each state-action pair)
        Q-table
        alphas_search (if searching_method = tmopson-sampling)
        betas_search (if searching_method = tmopson-sampling)

        Guessing table:
            alphas_guess (TS parameters for all trajectory)
            betas_guess (TS parameters for all trajectory)
            guess_q_table (Q table for state=trajectory, action=bet)

        Notice that we keep a register of how many times Bob did a state-action pair, which is useful in the case of fixed energy "static environment", and also to easily see exact convergence of the q-table

        """
        ######## q-table #####


        self.q_table = []
        self.n_table = []
        if self.method=="thompson-sampling":
            self.alphas_search=[]
            self.betas_search=[]
        for layer in range(self.layers):
            indexes = []
            for i in range(layer):
                indexes.append(2)
            for i in range(layer+1):
                indexes.append(len(self.actions))
            if self.algorithm == "UCB-eff":
                q_table_layer = np.ones(   tuple( indexes )  )*(self.layers+1)
            else:
                q_table_layer = np.zeros(   tuple( indexes )  )
            n_table_layer = np.zeros(   tuple( indexes )  )

            self.q_table.append(q_table_layer)
            self.n_table.append(n_table_layer)

            if self.method=="thompson-sampling":
                self.alphas_search.append(np.ones(tuple(indexes)))
                self.betas_search.append(np.ones(tuple(indexes)))

        if self.guessing_rule=="None":
            #Note: the (2, ) is to append the all possibles n_L (not considered in the q_table)
            self.guess_q_table = np.zeros((2,)+ self.q_table[-1].shape + (self.number_phases,))
            self.guess_visits_counter = np.zeros( (2,)+ self.q_table[-1].shape + (self.number_phases,))

            if self.method_guess == "thompson-sampling":
                self.alphas_guess = np.ones( (2,) + self.q_table[-1].shape + (self.number_phases,))
                self.betas_guess = np.ones( (2,)+ self.q_table[-1].shape + (self.number_phases,))


            elif self.method_guess == "ucb":
                self.guess_ucb_table = np.zeros((2,)+ self.q_table[-1].shape + (self.number_phases,))
        return



    def reset(self):
        """Reset agent state to dummy initial state (no partial observations yet)."""
        self.layer=0
        self.actions_index_did = []
        self.outcomes_observed = []
        self.actions_value_did=[]
        return

    def gather_outcome(self,outcome):
        """Put observed outcomes of photodetectors into memory"""
        self.outcomes_observed.append(outcome)
        return

    def select_action(self):
        """Given internal state of the agent, select next label of displacement"""
        if self.algorithm == "UCB-eff":
            actual_q = self.q_table[self.layer][ tuple(self.outcomes_observed) ][ tuple(self.actions_index_did)  ]
            action_index = np.where(actual_q == np.max(actual_q))[0]
            action, action_index = self.give_disp_value(action_index)
            self.actions_index_did.append(action_index)
            self.actions_value_did.append(action)
            return action_index, action

        # if (self.method=="ep-greedy")|(self.experiments_did<self.min_actions):
        if (self.method=="ep-greedy"):

            # if (self.method != "ep-greedy")&(self.experiments_did==0):
            # if (self.method != "ep-greedy")&(self.experiments_did==0):
            #
            #     print("Trying stage!")
            #     self.ep_saved = self.ep
            #     self.ep_method_saved = self.ep_method
            #     self.ep_method = "classics"
            #     self.ep = 1

            if self.ep_method=="exp-decay": #set to whatever otherwise
                self.ep = np.exp(-self.experiments_did/self.time_tau)
                if self.ep<self.min_ep:
                    self.ep=self.min_ep
            r = random.random()
            if (r< self.ep) | (self.ep==1):
                action_index = random.choice(self.action_indexes)
                action, action_index = self.give_disp_value(action_index)
                self.actions_index_did.append(action_index)
                self.actions_value_did.append(action)
                return action_index, action
            else:

                actual_q = self.q_table[self.layer][ tuple(self.outcomes_observed) ][ tuple(self.actions_index_did)  ]
                action_index = np.where(actual_q == np.max(actual_q))[0]
                action, action_index = self.give_disp_value(action_index)
                self.actions_index_did.append(action_index)
                self.actions_value_did.append(action)
                return action_index, action

        # elif (self.method == "ucb")&(self.experiments_did>=self.min_actions):
        elif (self.method == "ucb"):
            # if self.experiments_did == self.min_actions:
            #     self.ep = self.ep_saved
            #     self.ep_method = self.ep_method_saved
            n_visits = np.array(self.n_table[self.layer][tuple(self.outcomes_observed[:self.layer])][tuple(self.actions_index_did[:(self.layer+1)])])+1
            if self.ucb_method=="ucb1":
                ucb =np.sqrt(2* np.log(np.sum(n_visits))/ n_visits)
                # np.save("ucb1/ucb1"+str(self.experiments_did),ucb,allow_pickle=True)

            elif self.ucb_method=="ucb2":
                time = np.sum(n_visits)
                ucb = np.sqrt(2*np.log(1 + time*np.log(time)**2)/n_visits)
                # np.save("ucb2/ucb2"+str(self.experiments_did),ucb,allow_pickle=True)
            elif self.ucb_method == "ucb3":
                ucb = np.sqrt(2* np.log(np.sum(n_visits)))/ n_visits
            elif self.ucb_method == "ucb4":
                #https://arxiv.org/abs/1102.2490
                #I use c=0 for optimality according to the authors...
                #Notice i put [actions]
                qs = np.arange(.01,1,.01)
                ucb = np.zeros(len(n_visits))
                for actions in range(len(n_visits)):
                    to_max = []
                    for q in qs:
                        value_inside = n_visits[actions]*self.kl(self.q_table[self.layer][tuple(self.outcomes_observed)][tuple(self.actions_index_did)][actions], q)
                        if value_inside <= np.log(self.experiments_did+1):
                            to_max.append(value_inside)
                        else:
                            to_max.append(-1)
                    ucb[actions] = -self.q_table[self.layer][tuple(self.outcomes_observed)][tuple(self.actions_index_did)][actions] + max(to_max)
            else:
                print("Error in the ucb method! is either ucb1, ucb2 or ucb3")

            ucb_q_table = self.q_table[self.layer][tuple(self.outcomes_observed)][tuple(self.actions_index_did)] + ucb
            action_index = np.where(ucb_q_table == max( ucb_q_table ))[0]
            action, action_index = self.give_disp_value(action_index)
            self.actions_index_did.append(action_index)
            self.actions_value_did.append(action)

            return action_index, action

        # elif (self.method == "thompson-sampling")&(self.experiments_did>=self.min_actions):
        elif (self.method == "thompson-sampling"):
            if self.experiments_did == self.min_actions:
                self.ep = self.ep_saved
                self.ep_method = self.ep_method_saved
            # np.random.seed(datetime.now().microsecond()*int(np.random.random()))
            th = np.random.beta(self.alphas_search[self.layer][tuple(self.outcomes_observed[:self.layer])][tuple(self.actions_index_did[:(self.layer+1)])], self.betas_search[self.layer][tuple(self.outcomes_observed[:self.layer])][tuple(self.actions_index_did[:(self.layer+1)])]   )
            action_index = np.argmax(th)
            action, action_index = self.give_disp_value(action_index)
            self.actions_index_did.append(action_index)
            self.actions_value_did.append(action)
            return action_index, action

    def give_disp_value(self, label):
        """ Translates the label to the value of the displacement. """
        try: #this is because we use this method in two different situations: select label and also search for argmax for greedy_Q_prob
            if len(label)>1:
                label = random.choice(label)
            else:
                label = label[0]
        except Exception:
            # print("errorcito!!")
            pass
        action = self.actions[label]
        self.layer +=1
        return action, label




    def greedy_Q_prob(self):

        if self.guessing_rule == "None":
            if (self.method_guess == "thompson-sampling"):
                p=0
                if self.layers == 1:
                    l = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
                    b, l = self.give_disp_value(l)
                    for n1 in [0,1]:
                        aa= self.alphas_guess[tuple([n1,l0])]
                        bbb = self.betas_guess[tuple([n1,l0])]
                        ph = self.possible_phases[np.argmax((aa + bbb)/np.array(bbb))]
                        p+=(1-self.pflip)*self.P(ph*self.amplitude, b ,1, n1) + self.pflip*self.P(-ph*self.amplitude, b ,1, n1)
                    return p/self.number_phases

                elif self.layers == 2:
                    # l0 = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
                    # b0, l0 = self.give_disp_value(l0)
                    #
                    # l10 = np.where(self.q_table[1][0][l0,:] == np.max(self.q_table[1][0][l0,:]))[0]
                    # b10, l10 = self.give_disp_value(l10)
                    #
                    # l11 = np.where(self.q_table[1][1][l0,:] == np.max(self.q_table[1][1][l0,:]))[0]
                    # b11, l11 = self.give_disp_value(l11)

                    if self.method == "thompson-sampling":

                        al0 = self.alphas_search[0]
                        bl0 = self.betas_search[0]
                        means = np.array(al0)/np.array(al0+bl0)
                        l0 = np.where(means == np.max(means))[0]
                        b0, l0 = self.give_disp_value(l0)

                        al10 = self.alphas_search[1][0][l0,:]
                        bl10 = self.betas_search[1][0][l0,:]
                        means10 = np.array(al10)/np.array(al10+bl10)
                        l10 = np.where(means10 == np.max(means10))[0]
                        b10, l10 = self.give_disp_value(l10)

                        al11 = self.alphas_search[1][1][l0,:]
                        bl11 = self.betas_search[1][1][l0,:]
                        means11 = np.array(al11)/np.array(al11+bl11)
                        l11 = np.where(means11 == np.max(means11))[0]
                        b11, l11 = self.give_disp_value(l11)

                    else:
                        l0 = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
                        b0, l0 = self.give_disp_value(l0)

                        l10 = np.where(self.q_table[1][0][l0,:] == np.max(self.q_table[1][0][l0,:]))[0]
                        b10, l10 = self.give_disp_value(l10)

                        l11 = np.where(self.q_table[1][1][l0,:] == np.max(self.q_table[1][1][l0,:]))[0]
                        b11, l11 = self.give_disp_value(l11)

                    for n1,n2 in zip([0,0,1,1],[0,1,0,1]):
                        if n1==0:
                            beta2, label2 = b10, l10
                        else:
                            beta2, label2 = b11, l11
                        aa= self.alphas_guess[tuple([n1,n2,l0,label2])]
                        bbb = self.betas_guess[tuple([n1,n2,l0,label2])]
                        mph = (aa)/np.array(aa + bbb)
                        phl = np.where(mph == np.max(mph))[0]
                        try:

                            if len(phl)>1:
                                phl = random.choice(phl)
                            else:
                                phl = phl[0]
                        except Exception:
                            pass
                        ph = self.possible_phases[phl]
                        p+=(1-self.pflip)*self.P(ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(ph*self.amplitude, beta2 ,1/np.sqrt(2), n2) + (self.pflip*self.P(-ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(-ph*self.amplitude, beta2 ,1/np.sqrt(2), n2))
                    return p/self.number_phases
                else:
                    # return self.probability_going_Q_greedy_ml()
                    print("something went wrong, check the code out, i'm at line 347 of agent.py")
                    return .5
            else:
                p=0
                if self.layers == 1:
                    l = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
                    b, l = self.give_disp_value(l)
                    for n1 in [0,1]:
                        ph = np.where(self.guess_q_table[n1,l,:] == np.max(self.guess_q_table[n1,l,:]))[0]
                        if len(ph)>1:
                            ph = random.choice(ph)
                        else:
                            ph=ph[0]
                        ph = self.possible_phases[ph]
                        p+=(1-self.pflip)*self.P(ph*self.amplitude, b ,1, n1) +  self.pflip*self.P(-ph*self.amplitude, b ,1, n1)
                    return p/self.number_phases

                elif self.layers == 2:
                    l0 = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
                    b0, l0 = self.give_disp_value(l0)

                    l10 = np.where(self.q_table[1][0][l0,:] == np.max(self.q_table[1][0][l0,:]))[0]
                    b10, l10 = self.give_disp_value(l10)

                    l11 = np.where(self.q_table[1][1][l0,:] == np.max(self.q_table[1][1][l0,:]))[0]
                    b11, l11 = self.give_disp_value(l11)

                    for n1,n2 in zip([0,0,1,1],[0,1,0,1]):
                        if n1==0:
                            beta2, label2 = b10, l10
                        else:
                            beta2, label2 = b11, l11
                        ph = np.where(self.guess_q_table[n1,n2,l0,label2,:] == np.max(self.guess_q_table[n1,n2,l0,label2,:]))[0]
                        if len(ph)>1:
                            ph = random.choice(ph)
                        else:
                            ph=ph[0]

                        ph = self.possible_phases[ph]
                        p+=(1-self.pflip)*self.P(ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(ph*self.amplitude, beta2 ,1/np.sqrt(2), n2) + (self.pflip*self.P(-ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(-ph*self.amplitude, beta2 ,1/np.sqrt(2), n2))
                    return p/self.number_phases
                else:
                    # return self.probability_going_Q_greedy_ml()
                    return .5
        else:
            return self.probability_going_Q_greedy_ml()

    def probability_going_Q_greedy_ml(self):
        """
        //// deprecated and not tested (not used in our experiments any more..) ///
        returns the probability of success if you go greedy with respect to the Q-table
        at each step and guess by ML
        """

        if self.layers ==1:
            label = np.where(self.q_table[0] == np.max(self.q_table[0]))[0]
            displacement,label = self.give_disp_value(label)
            if self.guessing_rule == "Dolinar": #Notice i can just put return self.err_dolinar... they are the same (checked)
                p=0
                if displacement<0:
                    g = 1
                else:
                    g=0
                for n in [0,1]:
                    p+=self.P((-1)**(n+1+g)*self.amplitude, displacement,1,n)
                return p/2
            else:
                p=0
                for n in [0,1]:
                    p+=np.max([self.P(ph*self.amplitude, displacement,1,n) for ph in [-1,1]])
                return p/2

        else:

            dict = self.trajectory_dict.copy()
            disp=[]
            for ot in outcomes_universe(self.layers-1):
                if self.layers==2: ot=[ot]
                for layer in range(self.layers):
                    if layer ==0:
                        last_disp_labels=[]
                    else:
                        last_disp_labels.append(dict[str([int(x) for x in ot[:(layer-1)]])])
                    label = np.argmax(self.q_table[layer][tuple(ot[:(layer)])][tuple(last_disp_labels)])
                    dict[str([int(x) for x in ot[:layer]])] = label
            # ord=[] #just to check if the order is correct (importand for probability_error function)
            for layer in range(self.layers):
                self.layer=layer
                if layer==0:
                    disp.append(self.give_disp_value(dict["[]"])[0])
                    # ord.append("[]") #just to check if the order is correct (importand for probability_error function)
                else:
                    for ot in outcomes_universe(layer):
                        if layer==1: ot = [ot]
                        disp.append(self.give_disp_value(dict[str(list(ot))])[0]) ####BUG IS HERE!!!!!
                        # ord.append(ot) #just to check if the order is correct (importand for probability_error function)
            return 1-self.probability_error(disp)

    def q_learn(self, reward):
        """ Update of the Q-table.
            Note that we also update the parameters of the TS searching both for the guessing stage and also in the case of TS sampling for displacements.

         """
        if self.algorithm == "UCB-eff":
            self.QUCB_learn(reward)
            return
        else:

            self.experiments_did +=1
            for layer in range(self.layers):

                self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] +=1
                if self.learning_rate==0:
                    learning_rate = 1/self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])]
                else:
                    learning_rate=self.learning_rate

                if (layer==self.layers-1):
                    if self.guessing_rule == "None":

                        self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += learning_rate*( np.max(self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)]) -self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])
                        if self.method_guess == "thompson-sampling":
                            self.alphas_guess[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] += self.soft_ts*reward
                            self.betas_guess[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] += 1- (self.soft_ts*reward)
                        self.guess_visits_counter[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] +=1
                        times_guessed_here = self.guess_visits_counter[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess]
                        self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] += (reward - self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] )/times_guessed_here

                    else:
                        self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += learning_rate*( reward -self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])
                else:
                    self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += learning_rate*(   np.max( self.q_table[layer+1][tuple(self.outcomes_observed[:(layer+1)])][ tuple(self.actions_index_did[:(layer+1)])   ]   )   - self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])

                if self.method=="thompson-sampling":
                    if self.ts_method != "update_to_q":

                        self.alphas_search[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += self.soft_ts*(reward)
                        self.betas_search[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] +=  (1- self.soft_ts*reward)
                    else:
                        q_value = self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])]
                        self.alphas_search[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += self.soft_ts*(q_value)
                        self.betas_search[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] +=  (1- self.soft_ts*q_value)
            return

    def QUCB_learn(self, reward):
        #not in the paper
        self.experiments_did +=1
        for layer in range(self.layers):

            self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] +=1
            time = self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])]
            learning_rate = (self.layers+2)/(self.layers+1 +self.n_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])

            bt = self.c*np.sqrt(np.log(self.experiments_did)*(self.layers+1)**3*(self.iota)/time)

            if (layer==self.layers-1):
                self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += learning_rate*( bt + min(np.max(self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)]),self.layers+1) -self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])

                self.guess_visits_counter[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] +=1
                times_guessed_here = self.guess_visits_counter[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess]
                self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] += (reward - self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][self.guess] )/times_guessed_here

            else:
                self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])] += learning_rate*( bt+  min(np.max( self.q_table[layer+1][tuple(self.outcomes_observed[:(layer+1)])][ tuple(self.actions_index_did[:(layer+1)])   ]   ),self.layers+1)   - self.q_table[layer][tuple(self.outcomes_observed[:layer])][tuple(self.actions_index_did[:(layer+1)])])
        return





    def give_guess(self):
        """

        Giving the guess for the phase (choose one phase among the number_phases).
        For the binary case, using optimal displacements, optimal guessing rule (equals maximum likelihood)
        is equivalent to bet for parity of sum of outcomes (Dolinar).

        Notice that we keep self.guess in memory to update the guessing q_table """

        if self.guessing_rule == "Dolinar":
            if self.actions_value_did[0]<0:
                g=1
            else:
                g=0
            self.guess = (np.sum(self.outcomes_observed)+1+g)%2
            return self.guess

        elif self.guessing_rule == "None":
            # if (self.method_guess == "ep-greedy")|(self.experiments_did<self.min_actions):
            if (self.method_guess == "ep-greedy"):
                if random.random() < self.ep:
                    self.guess = random.choice([0,1])
                    return self.guess
                else:
                    self.guess = np.argmax(self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)])
                    if isinstance(self.guess, list) == True:
                        if len(self.guess)>1:
                            self.guess = random.choice(self.guess)
                        else:
                            self.guess = self.guess[0]
                    return self.guess
            # elif (self.method_guess == "ucb") & (self.experiments_did>=self.min_actions):
            elif (self.method_guess == "ucb"):
                if self.experiments_did == self.min_actions:
                    self.ep = self.ep_saved
                n_visits = np.array(self.guess_visits_counter[tuple(self.outcomes_observed)][tuple(self.actions_index_did)])+1
                if self.ucb_method == "ucb1":
                    ucb = np.sqrt(2*np.log(np.sum(n_visits)+1)/n_visits)
                elif self.ucb_method == "ucb2":
                    time = np.sum(n_visits)
                    ucb = np.sqrt(2*np.log(1 + time*np.log(time)**2)/n_visits)
                elif self.ucb_method == "ucb3":
                    ucb = np.sqrt(2* np.log(np.sum(n_visits) +1))/ n_visits

                elif self.ucb_method == "ucb4":
                    #https://arxiv.org/abs/1102.2490
                    #I use c=0 for optimality according to the authors...
                    #Notice i put [actions]
                    qs = np.arange(.01,1,.01)
                    ucb = np.zeros(len(n_visits))
                    for actions in range(len(n_visits)):
                        to_max = []
                        for q in qs:
                            value_inside = n_visits[actions]*self.kl( self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][actions], q)
                            if value_inside <= np.log(self.experiments_did+1):
                                to_max.append(value_inside)
                            else:
                                to_max.append(-1)
                        ucb[actions] = - self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)][actions] + max(to_max)

                ucb_q_table = self.guess_q_table[tuple(self.outcomes_observed)][tuple(self.actions_index_did)] + ucb
                self.guess = np.where(ucb_q_table == max(ucb_q_table))[0]
                if len(self.guess)>1:
                    self.guess = random.choice(self.guess)
                else:
                    self.guess = self.guess[0]
                if isinstance(self.guess, list) == True:
                    if len(self.guess)>1:
                        self.guess = random.choice(self.guess)
                    else:
                        self.guess = self.guess[0]
                # print("here", self.experiments_did, self.method_guess,self.ucb_method, self.actions_index_did, self.actions_value_did)

                return self.guess
            # elif (self.method_guess == "thompson-sampling") & (self.experiments_did>=self.min_actions):
            elif (self.method_guess == "thompson-sampling"):
                if self.experiments_did == self.min_actions:
                    self.ep=self.ep_saved
                theta = np.random.beta(self.alphas_guess[tuple(self.outcomes_observed)][tuple(self.actions_index_did)],self.betas_guess[tuple(self.outcomes_observed)][tuple(self.actions_index_did)])
                self.guess = np.argmax(theta)
                return self.guess

        elif self.guessing_rule == "max-likelihood": ##Careful, it's not tested!
            final=0
            for last_outcome in [0,1]:
                compare=[]
                compare_guess = [] #
                for phase in self.possible_phases:
                    term = 1
                    for j in range(self.layers-1):
                        r =  np.prod( np.sin(self.at[:j]))
                        t = np.cos(self.at[j])
                        term = term*self.P(phase*self.amplitude,self.actions_value_did[j],r*t, self.outcomes_observed[j])
                    r =  np.prod( np.sin(self.at[:(self.layers-1)]))
                    t = np.cos(self.at[self.layers-1])
                    term = term*self.P(phase*self.amplitude,self.actions_value_did[self.layers-1],r*t, last_outcome)
                    term_guess = term*self.P(phase*self.amplitude,self.actions_value_did[self.layers-1],r*t, self.outcomes_observed[self.layers-1])
                    compare_guess.append(term_guess)
                    compare.append(term)
                final+= np.max(compare)
            self.guess = np.argmax(compare_guess)
            # self.reward = np.max(final)/np.sum([ np.prod( [self.P(phase*self.amplitude, self.actions_value_did[mini_layer], np.cos(self.at[mini_layer])*np.prod(np.sin(self.at[:mini_layer])), self.outcomes_observed[mini_layer]) for mini_layer in range(self.layers-1)]) for phase in [-1,1]])
            return self.guess
        else:
            print("check out your guessing rule")
            return
