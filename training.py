import numpy as np
# import matplotlib.pyplot as plt
import agent
import environment
from tqdm import tqdm
# from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
import os
import shutil
from datetime import datetime
import glob
from misc import Record
import shutil
import basics
from misc import Kull
import random

class Experiment():
    """Simulation of discrimination experiment, repited number sates_wasted times.

        **amplitude: sqrt(intensity) of the states

        **layers: how many displacement/photodetectors Bob uses

        **resolution: resolution in discretized displacements Bob has to choose among

        **number_phases: How many phases Alice can pick. If N, the possible phases are the N square-roots of one.

        **bound_displacements: The displacements can be picked from [-bound_displacements, bound_displacements] (with the resolution given by resolution)

        **guessing_rule: At the end of each experiment, Bob has to guess the phase of the state (he may or may not know the energy)

                     - Max-likelihood (not tested):

                          1) When Bob knows the energy, he is given the reward according to the probability of success for that bunch of displacements (notice that it's not the probability of success, because he only has a sequence of outcomes. In this case, the reward is

                           np.max_\phases [ P(n1| \alpha, \beta1) P(n2| \alpha \beta2 (n1)) P(n3 | \alpha, \beta3(n2)) ] (constructed by Bob, see agent.q_learn() method) / np.prod(P(n_{1:(L-1) )   )

                       -Dolinar: Bob bets for the parity of the sum of the outcomes (in the case of number_phases=2, this is locally compatible with the maximum-likelihood strategy, which is always optimal), in the region of displacements that maximizes the probability of sucess in discriminating the two states.

                       - None: Bob knows nothing, and chooses his guess according to a Thompson Sampling search, updating his knowledge at each experiment


        **searching_method: How Bob decides to choose the next action. Notice that all these methods works in each state action pair, tabular case. To-do: extend this to approximate case.
                        1) [string] ep-greedy [string] : performes ep-greedy search (ruled by ep, next method)
                        2) [string] ucb [string] : uses Hoeffdrings inequality to perform upper confidence bound searching
                        3) [string] htompson-sampling [sting] : performs Thompson Sampling

        ** ucb_method: if searching method="ucb", you have different variants of the algorithm. The first one is heuristic from this work, and seems to work fine:
            "ucb1: UCB = sqrt(2*time/N)

            "ucb2" UCB= sqrt(f(t)/N) with f(t) = log(1 + tlog(t)) (see https://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/)

            "ucb3": (heuristic from here) UCB= sqrt(2* time)/N  Comment. Our heuristic method clearly favours exploitation

        **states_wasted: how many discrimination rounds Alice and Bob make, each round consisting on Alice picking a phase  (according to some prior) and sending the coherent state to Bob.

        ** soft_ts: \alpha -> \alpha + soft_ts *reward
                    \beta -> \beta + soft_ts*(1-rewad∫)

                    default is 1 (usual thompson sampling)

                    the reason i put this is to delay the exploitation period with TS

        ** std: suppose now Alice has a gaussian error in the energy... (set to 0 means no errors)

        ** ep_method: expoential decay epsilon greedy

        ** strawberry_fields: if "ON", simulates the outcome with the strawberryfields engine developed by xanadu

        ** efficiency: related to dark_counts (if 1 then perfect detectors)

        ** method_guess: how to choose the guessing rule to be learned: thompson-sampling, ep-greedy, ucb

        ** min_ep: minimum epsilon (for ep-greedy searching)

        ** time_tau: time decaying in the exponential ep-greedy

        **efficient_time: if take 10 data points from the last decade: between 0 and 10 i take 1, between 10 and 100 take 10, etc...

        ** pflip: probability that the true phase filps

        ** algorithm: if "UCB-eff", implements algorithm of is Q-learning provably efficient?
        ** strange_factor_ucbeff is the c that appears in the bound
        ** prob_eff is the probability of the regret being above...

        ** save_tables: if True then I save q-tables, guessing tables, etc... the evolution even! it's a lot lot lot of data
        -------------------------------------

            The agents retrain by default, saving in Experiment() object a dictionary called bobs, where each Bob object is stored with its trained q-table and all pertinent data.
        -------------------------------------

    """

    def __init__(self,  amplitude=.4, layers=2, resolution=0.1, number_phases=2, bound_displacements=1, guessing_rule="None", searching_method="ucb", ep=0.01, ucb_method="ucb1", states_wasted=10**4, soft_ts=1, soft_ts_guess=1, std_energy=0, ep_method ="exp-decay", strawberry_fields="OFF",efficiency=1,method_guess="undefined", min_ep = 0.01, time_tau=1000, efficient_time=True, pflip = 0, algorithm="standard", strange_factor_ucbeff=1, prob_eff=0.01, save_tables="Completely-False", ts_method= "std"):

        self.info = " amplitude:" + str(amplitude) + "\nlayers: "+str(layers) + "\n number_phases: " + str(number_phases) + "\n search_method: " +str(searching_method)+ "\n resolution: " +str( resolution) + "\n bound_displacements: "+str(bound_displacements) + "\n guessing_rule: " +str( guessing_rule) + "\n method_guess : "+str(method_guess) + "dark counts rate (efficiency): " + str(efficiency) + "\n Phase flip probability: "+ str(pflip) + " \n ************  \n ************ \n " +  "epsilon: "+str(ep) + "\n epsilon_greedy method: "+ str( ep_method) +  "\n min_ep : "+str(min_ep) + "\n time_tau (exp decay ep-greedy): "+str(time_tau) + "\n \n  **** UCB DETAILS *** \n \n " + " ucb_method: " + str(ucb_method) + "**** thomspon sampling details: " + "\n soft_ts: " + str(soft_ts) + "\n \n **** TS DETAILS \n ts_method (if update towards q instead of reward): " + str(ts_method) + "\nsoft_ts: "+str(soft_ts)

        if algorithm == "UCB-eff":
            self.info = "\n \n ********** \n \n ###### ALGORITHM OF IS Q-LEARNING PROVABLY EFFICIENT:" + "c : " + str(strange_factor_ucbeff) + "\n prob_eff: (with 1-p the regret is at most O(\sqrt{T})): " +str(prob_eff)


        self.amplitude = amplitude
        self.layers = layers
        self.resolution = resolution
        self.number_phases=number_phases
        self.bound_displacements = bound_displacements
        self.states_wasted=states_wasted

        self.searching_method = searching_method
        self.guessing_rule = guessing_rule
        self.efficiency = efficiency
        self.pflip = pflip
        self.efficient_time = efficient_time

        self.ep = ep
        self.min_ep = min_ep
        self.time_tau = time_tau

        self.bobs = {}
        self.ep_method = ep_method
        self.ucb_method = ucb_method
        self.method_guess=method_guess

        self.soft_ts = soft_ts
        self.ts_method = ts_method

        self.std_energy = std_energy

        self.strawberry_fields = strawberry_fields

        bb = basics.Basics(amplitude=self.amplitude, layers=1,bound_displacements=self.bound_displacements, resolution=self.resolution)
        self.homodyne_limit = bb.homodyne()
        del bb

        self.opt_kenn = np.genfromtxt("bounds_optimals_and_limits/kennedy_probs/"+str(np.round(self.amplitude,2))+".csv", delimiter=",")
        self.opt_2l = np.genfromtxt("bounds_optimals_and_limits/2layers_probs/"+str(np.round(self.amplitude,2))+".csv", delimiter=",")
        # self.opt_kenn_resolution = self.compute_optimal_kenn()
        self.optimal_value = 0
        # if (self.layers == 1)&(self.resolution==0.1)&(self.bound_displacements==1)&(np.round(self.amplitude,2)==self.amplitude):
        #     self.optimal_value = np.load("bounds_optimals_and_limits/1layers_probs_resolution0.1/"+str(np.round(self.amplitude,2))+".npy")
        #     self.cte_LR = 1505.5002588209475
        if self.layers == 2:
            self.opt_2l_resolution01 = np.load("bounds_optimals_and_limits/2layers_probs_resolution0.1/"+str(np.round(self.amplitude,2))+".npy")
            self.optimal_value = self.opt_2l_resolution01
            self.cte_LR = 1505.5002588209475 #(the one for bandits...)

        elif self.layers==1:
            #for the bandit scenarii...
            b = basics.Basics(amplitude=self.amplitude, layers=1,bound_displacements=self.bound_displacements, resolution=self.resolution)
            b.define_actions()
            dol_prob= [b.err_dolinar(disp) for disp in b.actions]
            self.optimal_value =max(dol_prob)

            reg_coeff=0
            for p in dol_prob:
                if p!=self.optimal_value:
                    reg_coeff += (self.optimal_value -p)/Kull(p,self.optimal_value)

            self.cte_LR = reg_coeff
        else:
            self.optimal_value = "Compute it with DP!"
        self.algorithm = algorithm
        self.strange_factor_ucbeff = strange_factor_ucbeff
        self.prob_eff = prob_eff

        self.times_bkp_tables = []
        for k in range(1,int(np.log10(self.states_wasted))): #the +1 is just in case...
            self.times_bkp_tables = np.append(self.times_bkp_tables, np.arange(10**k, 10**(k+1), 10**k))

        if self.efficient_time == True:
            self.times_saved = self.times_bkp_tables
        else:
            self.times_saved = np.arange(1,self.states_wasted+1)
        self.save_tables = save_tables

        # #Computed for 0.1R_1Bound displcement, 2 phases, 1Layer

    def LRBound(self, time):
        return self.cte_LR*np.log(time)

    def compute_optimal_kenn(self):
        f = basics.Basics(layers=1, resolution=self.resolution, amplitude=self.amplitude, bound_displacements=self.bound_displacements, efficiency=self.efficiency, pflip=self.pflip)
        f.define_actions()
        self.optimal_kenn_resolution= 1-np.min(f.err_kennedy(f.actions))
        np.save("bounds_optimals_and_limits/1layers_probs_resolution0.1/" + str(self.amplitude),self.optimal_kenn_resolution)

        del f
        return

    def compute_optimal_2l(self):
        f = basics.Basics(layers=2, resolution=self.resolution, amplitude=self.amplitude, bound_displacements=self.bound_displacements, efficiency=self.efficiency, pflip=self.pflip)
        f.define_actions()
        probs=[]
        for b1 in f.actions:
            for b20 in f.actions:
                for b21 in f.actions:
                    probs.append(f.probability_error([b1,b20,b21]))
        del f
        self.optimal_2l_resolution = 1-np.min(probs)
        del probs
        np.save("bounds_optimals_and_limits/2layers_probs_resolution0.1/" + str(self.amplitude),self.optimal_2l_resolution)
        return

    def train(self, number_bobs=12):
        self.number_bobs = str(number_bobs)
        self.average_bobs(number_bobs=number_bobs)
        self.save_data()
        return


    def average_bobs(self,number_bobs=12):
        self.number_bobs=number_bobs
        print("Training: \n searching policy: "+str(self.searching_method) +"\n guessing policy: "+str(self.guessing_rule) + " - "+ str(self.method_guess) +  "\n number of agents: "+str(number_bobs) )
        if not os.path.exists("temporal_data"):
            os.makedirs("temporal_data")

        if number_bobs > 1:
            p=[mp.Process(target=self.training_bob, args=(str(i),)) for i in range(number_bobs)]
            for pp in p:
                pp.start()
            for pp in p:
                pp.join()
            results = self.collect_results(number_bobs)
            return results
        else:
            return self.training_bob("No-id")


    def training_bob(self, bob_id="No-id"):
        """
        Main method, where the POMDP occurs. This is paralellised - if desired - among different cores.
        The learning of optimal displacements is done via Q-learning,
        as can be seen at Agent class,
        from the agent.py file
        """


        if bob_id!="No-id": #This is because there are problems with multiprocessing and the random number generator otherwise (all bobs the same, twins bobs xD)
            np.random.seed(int(bob_id)*datetime.now().microsecond)
            random.seed(int(bob_id)*datetime.now().microsecond)

        bob = agent.Agent(layers = self.layers, resolution=self.resolution, number_phases =self.number_phases, bound_displacements = self.bound_displacements, guessing_rule = self.guessing_rule, searching_method=self.searching_method, ep=self.ep, ep_method=self.ep_method, ucb_method=self.ucb_method, soft_ts = self.soft_ts, efficiency=self.efficiency,method_guess=self.method_guess, min_ep=self.min_ep, time_tau = self.time_tau, pflip=self.pflip, algorithm=self.algorithm, strange_factor_ucbeff=self.strange_factor_ucbeff, prob_eff=self.prob_eff, ts_method=self.ts_method)

        bob.ep=self.ep
        bob.probability_success_greedy_q=[]
        bob.cumulative = 0
        bob.cumulative_reward_evolution = []

        q_tables = []
        n_tables = []
        q_guess_tables = []
        n_guess_tables = []

        times_being=[]
        regret_sums=[]
        cc=0
        # if bob.guessing_rule == "None":
        #
        #     if bob.method_guess == "thompson-sampling":
        #         alphas_guess = []
        #         betas_guess = []
        #     if bob.searching_method == "thompson-sampling":
        #         alphas_disp = []
        #         betas_disp = []

        alice = environment.Environment(amplitude = self.amplitude, layers= self.layers, resolution = self.resolution, number_phases = self.number_phases, bound_displacements = self.bound_displacements, std=self.std_energy, efficiency=self.efficiency, pflip=self.pflip)

        print("training for ", self.states_wasted)
        for k in tqdm(range(1,self.states_wasted+1)):
            alice.pick_phase()
            bob.reset()

            for layer in range(bob.layers):
                action_index, action = bob.select_action()
                outcome = alice.give_outcome(action, layer)
                bob.gather_outcome(outcome)

            guess = bob.give_guess()
            reward = alice.give_reward(guess)
            bob.q_learn(reward)
            if (self.layers==1)&(self.guessing_rule=="Dolinar"):
                p = bob.err_dolinar(bob.actions_value_did[0])
                cc += p
                regret_sums.append(cc)
                # print(k, cc)
            else:
                pass
                #prob_behaviour_policy.append(.5) #it's not that trivial to compute the success probability of behaviour policy, try to think on ep-greedy or TS... a complete mess! Notice otherwise that the UCB-Qlearning from paper Is Q-learning provably efficeint? makes it easier: just go greedy with respect to Q-table + UCB (in the way we implement the algorithm this prob_behaviour_policy would equal the success probability o behaviour policy...)

            bob.cumulative+=reward
            if k in self.times_saved:
                times_being.append(k)
                bob.cumulative_reward_evolution.append(bob.cumulative)
                bob.probability_success_greedy_q.append(bob.greedy_Q_prob())

            if (k in self.times_bkp_tables)&(self.save_tables==True):
                q_tables.append(bob.q_table)
                n_tables.append(bob.n_table)
                if bob.guessing_rule == "None":
                    q_guess_tables.append(bob.guess_q_table)
                    n_guess_tables.append(bob.guess_visits_counter)

                #### It's too much data to save for the parameters of thompson sampling...
                # if bob.method_guess == "thompson-sampling":
                #     alphas_guess.append(bob.alphas_guess)
                #     betas_guess.append(bob.betas_guess)
                # if bob.searching_method == "thompson-sampling":
                #     alphas_disp.append(bob.alphas_search)
                #     betas_disp.append(bob.betas_search)

        learning_curves = [self.times_saved, bob.cumulative_reward_evolution, bob.probability_success_greedy_q]
        if (self.layers == 1)&(self.guessing_rule=="Dolinar"):
            # regret = self.optimal_value*np.arange(1,len(self.times_saved)+1) - regret_sums
            regret = self.optimal_value*np.arange(1,len(self.times_saved)+1) - learning_curves[1]

            # learning_curves.append(regret)
            learning_curves.append(regret)

            # learning_curves.append(self.compute_regret(prob_behaviour_policy))
        else:
            if self.algorithm=="UCB-eff":
                learning_curves.append(range(learning_curves[2])) #but this would only be the regret for the case in which we have self.algorithm = "UCB-eff",, or the 0-greedy policy.
            else:
                regret = self.optimal_value*np.arange(1,len(self.times_saved)+1) - learning_curves[2]
                learning_curves.append(regret)

        if bob_id != "No-id":
            np.save("temporal_data/"+str(bob_id)+"data_learning_curve",np.array(learning_curves))
            if self.save_tables==True:
                np.save("temporal_data/"+str(bob_id)+"QsDISP_evolution",q_tables)
                np.save("temporal_data/"+str(bob_id)+"NsDISP_evolution",n_tables)
                if bob.guessing_rule == "None":
                    np.save("temporal_data/"+str(bob_id)+"qsGUESS_evolution",q_guess_tables)
                    np.save("temporal_data/"+str(bob_id)+"NsGUESS_evolution",n_guess_tables)
            return
        else:
            np.save("locallearning_curves",np.array(learning_curves))

            np.save("temporal_data/learning_curves",np.array(learning_curves))
            np.save("temporal_data/stds",np.array([])) #just to use load_data method for all situations (in showing.py we omit this)
            np.save("temporal_data/minimax",np.array([]))

            if self.save_tables==True:
                np.save("temporal_data/0QsDISP_evolution", q_tables)
                np.save("temporal_data/0NsDISP_evolution", n_tables)
                if bob.guessing_rule == "None":
                    np.save("temporal_data/0qsGUESS_evolution",q_guess_tables)
                    np.save("temporal_data/0NsGUESS_evolution",n_guess_tables)
            return

    def collect_results(self, number_bobs):
        data_agents = {}

        if self.save_tables==True:
            qtable_avgs = np.load("temporal_data/"+str(0)+"QsDISP_evolution.npy", allow_pickle=True)
            ntable_avgs = np.load("temporal_data/"+str(0)+"NsDISP_evolution.npy", allow_pickle=True)
            if self.guessing_rule == "None":
                qtableGuess_avgs = np.load("temporal_data/"+str(0)+"qsGUESS_evolution.npy", allow_pickle=True)
                ntableGuess_avgs = np.load("temporal_data/"+str(0)+"NsGUESS_evolution.npy", allow_pickle=True)

        #MEAN AND STD OF EACH QUANTITY

        tot_ep = len(self.times_saved) #self.times_saved defined in __init__
        r_cumulative = np.zeros(tot_ep)
        pr_gre = np.zeros(tot_ep)
        r_cumulative_std = np.zeros(tot_ep)
        pr_gre_std = np.zeros(tot_ep)
        regret = np.zeros(tot_ep)
        regret_std = np.zeros(tot_ep)

        for i in range(number_bobs):
            data_agents[str(i)] = np.load("temporal_data/"+str(i)+"data_learning_curve.npy",allow_pickle=True)
            r_cumulative += data_agents[str(i)][1]
            pr_gre += data_agents[str(i)][2]
            regret += data_agents[str(i)][3]

        r_cumulative = r_cumulative/number_bobs
        pr_gre = pr_gre/number_bobs
        regret = regret/number_bobs


        for i in range(number_bobs):
            r_cumulative_std += np.square(data_agents[str(i)][1]-r_cumulative)
            pr_gre_std += np.square(data_agents[str(i)][2]- pr_gre)
            regret_std += np.square(data_agents[str(i)][3] - regret)

        r_cumulative_std = np.sqrt(r_cumulative_std/(number_bobs-1))
        pr_gre_std = np.sqrt(pr_gre_std/(number_bobs-1))
        regret_std = np.sqrt(regret_std/(number_bobs-1))


        #### MIN, MAX OF EACH QUANTITY
        min_r_cumulative, max_r_cumulative = [], []
        min_pr_gre, max_pr_gre = [], []
        min_regret, max_regret = [], []
        for index_time in range(len(self.times_saved)):
            # for i in range(number_bobs):
                # print(i,index_time, data_agents[str(i)][2][index_time])

            min_r_cumulative.append(min([data_agents[str(i)][1][index_time] for i in range(number_bobs)]))
            max_r_cumulative.append(max([data_agents[str(i)][1][index_time] for i in range(number_bobs)]))

            min_pr_gre.append(min([data_agents[str(i)][2][index_time] for i in range(number_bobs)]))
            max_pr_gre.append(max([data_agents[str(i)][2][index_time] for i in range(number_bobs)]))

            min_regret.append(min([data_agents[str(i)][3][index_time] for i in range(number_bobs)]))
            max_regret.append(max([data_agents[str(i)][3][index_time] for i in range(number_bobs)]))



        if self.save_tables==True:
                        ########### SAVING TABLES PART #####
                        ########### SAVING TABLES PART #####
                        ########### SAVING TABLES PART #####

            for i in range(1,number_bobs):
                qs = np.load("temporal_data/"+str(i)+"QsDISP_evolution.npy", allow_pickle=True)
                ns = np.load("temporal_data/"+str(i)+"NsDISP_evolution.npy", allow_pickle=True)

                if self.guessing_rule == "None":
                    qsg = np.load("temporal_data/"+str(i)+"qsGUESS_evolution.npy", allow_pickle=True)
                    nsg = np.load("temporal_data/"+str(i)+"NsGUESS_evolution.npy", allow_pickle=True)

                for ind, qst in enumerate(qs):
                    qtable_avgs[ind] += qst
                    ntable_avgs[ind] += ns[ind]
                    if self.guessing_rule == "None":
                        qtableGuess_avgs[i] += qsg[ind]
                        ntableGuess_avgs[i] += nsg[ind]
            #

            for i in range(len(qtable_avgs)):
                qtable_avgs[i] = qtable_avgs[i]/number_bobs
                ntable_avgs[i] = ntable_avgs[i]/number_bobs
                if self.guessing_rule == "None":
                    qtableGuess_avgs[i] = qtableGuess_avgs[i]/ number_bobs
                    ntableGuess_avgs[i] = ntableGuess_avgs[i]/number_bobs


            np.save("temporal_data/q_disp_avg_evolution", qtable_avgs, allow_pickle=True)
            np.save("temporal_data/n_disp_avg_evolution", ntable_avgs, allow_pickle=True)
            if self.guessing_rule == "None":
                np.save("temporal_data/q_guess_avg_evolution", qtableGuess_avgs, allow_pickle=True)
                np.save("temporal_data/n_guess_avg_evolution", ntableGuess_avgs, allow_pickle=True)

                        ########### SAVING TABLES PART #####
        learning_curves = [self.times_saved, r_cumulative, pr_gre, regret]
        stds = [r_cumulative_std, pr_gre_std, regret_std]
        min_max = [min_r_cumulative, max_r_cumulative, min_pr_gre, max_pr_gre, min_regret, max_regret]
        np.save("temporal_data/learning_curves", learning_curves)
        np.save("temporal_data/stds", stds)
        np.save("temporal_data/minimax", min_max)
        return




    def save_data(self):
        name_folder = str(self.layers)+"L"+str(self.number_phases)+"PH"+str(self.resolution)+"R"
        self.number_run = Record(name_folder).number_run
        print("saving the results at ", os.getcwd())
        files = ["../../temporal_data/q_guess_avg_evolution.npy","../../temporal_data/n_guess_avg_evolution.npy", "../../temporal_data/q_disp_avg_evolution.npy","../../temporal_data/n_disp_avg_evolution.npy","../../temporal_data/learning_curves.npy", "../../temporal_data/minimax.npy", "../../temporal_data/stds.npy"]
        if self.save_tables == True:
            table_files = ["../../temporal_data/0QsDISP_evolution.npy","../../temporal_data/0NsDISP_evolution.npy", "../../temporal_data/0qsGUESS_evolution.npy","../../temporal_data/0NsGUESS_evolution.npy"]
            for tf in table_files:
                files.append(tf)

        for file in list(glob.glob('../../temporal_data/*')):
            if file in files:
                if file not in ["../../temporal_data/learning_curves.npy",  "../../temporal_data/minimax.npy", "../../temporal_data/stds.npy"]:
                    shutil.move(file,os.getcwd()+"/tables")
                else:
                    shutil.move(file,os.getcwd())

        shutil.rmtree("../../temporal_data")

        with open("info_run.txt", "w") as f:
            f.write(self.info + "\n **** number_bobs: "+str(self.number_bobs))
            f.close()


        os.chdir("../..")
        print(os.getcwd())
        return

    def load_data(self, run="", tables=False):
        name_folder = str(self.layers)+"L"+str(self.number_phases)+"PH"+str(self.resolution)+"R"
        print(name_folder)
        try:
            os.chdir(name_folder)
        except Exception:
            print("You have no data for this configuration..." + str(name_folder) + "\n Please make sure to train the agent(s) and then grab you'll be able to grab_data")
            return

        self.results = np.load(str(run)+"/learning_curves.npy", allow_pickle=True)
        self.stds = np.load(str(run)+"/stds.npy", allow_pickle=True)
        self.minimax = np.load(str(run)+"/minimax.npy", allow_pickle=True)

        with open(str(run)+"/info_run.txt", "r") as f:
            self.info = f.read()
        print(str(run))
        if tables==True:

            self.q_table_evolution = np.load(str(run)+"/tables/0QsDISP_evolution.npy", allow_pickle=True)
            self.n_table_evolution = np.load(str(run)+"/tables/0NsDISP_evolution.npy", allow_pickle=True)
            self.q_table_guess_evolution = np.load(str(run)+"/tables/0qsGUESS_evolution.npy", allow_pickle=True)
            self.n_table_guess_evolution = np.load(str(run)+"/tables/0NsGUESS_evolution.npy", allow_pickle=True)
            print("successfully loaded the tables :)")
            # except Exception:
            #     print("Error: make sure save_tables=True when you run the program (it's an option of training.Experiment())")
        os.chdir("..")
        return

    # def compute_regret(self, data):
    #     #deprecated .. not useful because takes a lot of tiime...
    #     regrets = []
    #     m=len(data)
    #     for index, time in enumerate(self.times_saved):
    #         if index%(m/10) == 0:
    #             print(index, m)
    #         regrets.append((time)*self.optimal_value - np.sum(data[:(index+1)])) #notice if a = [1,2,3] a[:4] is a ;)
    #     return regrets


#######
