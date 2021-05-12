import numpy as np
import agent
import environment
from tqdm import tqdm
import multiprocessing as mp
import os
import shutil
from datetime import datetime
import glob
from misc import Record
import shutil
import basics
import random

class Experiment():

    def __init__(self,  amplitude=.4, layers=2, n_actions=10, number_phases=2,
                bound_displacements=1, states_wasted=10**4,
                ep_method ="exp-decay", min_ep = 0.01,ep=0.01,
                tau_ep=1000, searching_method="ep_greedy", experiment_label="", channel={}):

        self.info = f"amplitude: {amplitude}\n" \
                f"layers: {layers}\n"\
                f"n_actions per layer: {n_actions}\n" \
                f"n_phases: {number_phases}\n" \
                f"states_wasted: {states_wasted }\n" \


        self.amplitude = amplitude
        self.layers = layers
        self.n_actions = n_actions
        self.number_phases=number_phases
        self.bound_displacements = bound_displacements
        self.states_wasted=states_wasted
        self.searching_method = searching_method

        self.ep = ep
        self.min_ep = min_ep
        self.tau_ep = tau_ep

        self.bobs = {}
        self.ep_method = ep_method


        self.saving_times = []
        for k in range(1,int(np.log10(self.states_wasted))): #the +1 is just in case...
            self.saving_times = np.append(self.saving_times, np.arange(10**k, 10**(k+1), 10**k))

        self.experiment_label = str(self.layers)+"L_"+str(self.n_actions)+"a"+experiment_label
        self.channel = channel

    def train(self, number_bobs=12):
        self.number_bobs = str(number_bobs)
        self.info+=f"Number of Bobs: {number_bobs }\n" 
        self.average_bobs(number_bobs=number_bobs)
        self.save_data()
        return


    def average_bobs(self,number_bobs=12):
        self.number_bobs=number_bobs
        info_training = f"Number of agents: {number_bobs}\n"
        print(info_training)

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
            return self.training_bob(0)


    def training_bob(self, bob_id):

        random.seed(int(bob_id))

        bob = agent.Agent(amplitude = self.amplitude,layers = self.layers, n_actions=self.n_actions,
                    bound_displacements = self.bound_displacements, searching_method= self.searching_method,
                    ep=self.ep, ep_method=self.ep_method,  min_ep=self.min_ep, tau_ep = self.tau_ep, channel=self.channel)

        bob.probability_success_greedy_q=[]

        bob.cumulative = 0
        bob.cumulative_reward_evolution = []

        q_tables = []
        n_tables = []
        q_guess_tables = []
        n_guess_tables = []

        times_being=[]

        alice = environment.Environment(amplitude = self.amplitude, layers= self.layers, channel=self.channel)


        bob.cumulative_reward_evolution = []
        bob.probability_success_greedy_q= []

        for k in tqdm(range(1,self.states_wasted+1)):
            bob.reset()
            alice.reset()

            alice.pick_phase()
            alice.act_channel()

            for layer in range(bob.layers):
                action_index, action = bob.select_action()
                bob.layer +=1
                outcome = alice.give_outcome(action, layer)
                bob.gather_outcome(outcome)

            guess = bob.give_guess()
            reward = alice.give_reward(guess)

            bob.q_learn(reward)

            bob.cumulative+=reward
            if k in self.saving_times:
                times_being.append(k)
                bob.cumulative_reward_evolution.append(bob.cumulative)
                if (self.layers<3):
                    bob.probability_success_greedy_q.append(bob.greedy_Q_prob())

            # if self.save_tables==True:
            #     q_tables.append(bob.q_table)
            #     n_tables.append(bob.n_table)
            #     if bob.guessing_rule == "None":
            #         q_guess_tables.append(bob.guess_q_table)
            #         n_guess_tables.append(bob.guess_visits_counter)

        learning_curves = [self.saving_times, bob.cumulative_reward_evolution, bob.probability_success_greedy_q]

        os.makedirs("temporal_data",exist_ok=True)
        np.save("temporal_data/"+str(bob_id)+"data_learning_curve",np.array(learning_curves))
        # np.save("temporal_data/"+str(bob_id)+"tables",np.array([bob.q_table,bob.guess_q_table,bob.n_table, bob.guess_n_table], allow_pickle=True)
        np.save("temporal_data/"+str(bob_id)+"qtable",bob.q_table, allow_pickle=True)
        np.save("temporal_data/"+str(bob_id)+"nables",bob.n_table, allow_pickle=True)
        np.save("temporal_data/"+str(bob_id)+"qguesstable",bob.guess_q_table, allow_pickle=True)
        np.save("temporal_data/"+str(bob_id)+"nguesstable",bob.guess_n_table, allow_pickle=True)

        return

    def collect_results(self, number_bobs):
        data_agents = {}

        tot_ep = len(self.saving_times)
        r_cumulative = np.zeros(tot_ep)
        pr_gre = np.zeros(tot_ep)

        r_cumulative_std = np.zeros(tot_ep)
        pr_gre_std = np.zeros(tot_ep)


        for i in range(number_bobs):
            data_agents[str(i)] = np.load("temporal_data/"+str(i)+"data_learning_curve.npy",allow_pickle=True)
            r_cumulative += data_agents[str(i)][1]
            pr_gre += data_agents[str(i)][2]

        r_cumulative = r_cumulative/number_bobs
        pr_gre = pr_gre/number_bobs

        if number_bobs>1:

            for i in range(number_bobs):
                r_cumulative_std += np.square(data_agents[str(i)][1]-r_cumulative)
                pr_gre_std += np.square(data_agents[str(i)][2]- pr_gre)

            r_cumulative_std = np.sqrt(r_cumulative_std/(number_bobs-1))
            pr_gre_std = np.sqrt(pr_gre_std/(number_bobs-1))


        #### MIN, MAX OF EACH QUANTITY
        min_r_cumulative, max_r_cumulative = [], []
        min_pr_gre, max_pr_gre = [], []

        for index_time in range(len(self.saving_times)):

            min_r_cumulative.append(min([data_agents[str(i)][1][index_time] for i in range(number_bobs)]))
            max_r_cumulative.append(max([data_agents[str(i)][1][index_time] for i in range(number_bobs)]))

            min_pr_gre.append(min([data_agents[str(i)][2][index_time] for i in range(number_bobs)]))
            max_pr_gre.append(max([data_agents[str(i)][2][index_time] for i in range(number_bobs)]))



        learning_curves = [self.saving_times, r_cumulative, pr_gre]
        stds = [r_cumulative_std, pr_gre_std]
        min_max = [min_r_cumulative, max_r_cumulative, min_pr_gre, max_pr_gre]
        np.save("temporal_data/learning_curves", learning_curves)
        np.save("temporal_data/stds", stds)
        np.save("temporal_data/minimax", min_max)
        return


    def save_data(self):
        name_folder = self.experiment_label
        self.number_run = Record(name_folder).number_run
        print("saving the results at ", os.getcwd())
        files = ["../../temporal_data/q_guess_avg_evolution.npy","../../temporal_data/n_guess_avg_evolution.npy", "../../temporal_data/q_disp_avg_evolution.npy","../../temporal_data/n_disp_avg_evolution.npy","../../temporal_data/learning_curves.npy", "../../temporal_data/minimax.npy", "../../temporal_data/stds.npy"]

        for file in list(glob.glob('../../temporal_data/*')):
            if file in files:
                if file in ["../../temporal_data/learning_curves.npy",  "../../temporal_data/minimax.npy", "../../temporal_data/stds.npy"]:
                    shutil.move(file,os.getcwd()+"/tables")
                else:
                    shutil.move(file,os.getcwd())

        shutil.rmtree("../../temporal_data")

        with open("info_run.txt", "w") as f:
            f.write(self.info + "\n **** number_bobs: "+str(self.number_bobs))
            f.close()

        os.chdir("../..")
        # os.makedirs("reuslts",exist_ok=True)
        # os.system("mv {} results".format(name_folder))
        return


if __name__ == "__main__":

    ep = 0.3
    layers = 2
    n_actions = 10
    searching_method = "ep-greedy"
    bound_displacements = 1
    total_episodes = 5*10**5
    tau_ep=100
    ep_method="normal"
    nbobs=1
    min_ep=0.01

    experiment_label="_LC"
    channel = {"class":"compound_lossy", "params":[.5,0.01]}


    exper = Experiment(searching_method = searching_method, layers=layers, ep=ep,n_actions=n_actions,
        bound_displacements=bound_displacements, states_wasted=total_episodes, ep_method= ep_method, tau_ep=tau_ep,min_ep=min_ep,
         experiment_label=experiment_label, channel=channel)
    #
    # exper.training_bob(1)
    exper.average_bobs(8)
    exper.collect_results(nbobs)
    exper.save_data()





#
