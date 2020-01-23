import training
import gc
from showing import ploting
import pickle
from misc import save_obj, filter_keys
#In this program we compute the learning curves for the bandit problem in the Kennedy case, taking Dolinar's guessing rule.

#Each arm is a possible displacement.

class MegaFrontEnd():
    def __init__(self, layers=2, resolution=0.1, bound_displacements=1, guessing_rule="None", efficient_time = False):
        self.layers = layers
        self.guessing_rule = guessing_rule
        self.resolution = resolution
        self.bound_displacements = bound_displacements
        self.efficient_time = efficient_time
        # self.methods_to_run = ["ep-greedy", "exp-ep-greedy", "ucb", "thompson-sampling"]

    def single_run(self, total_episodes=10**2, bob=1):
        dict = {}
        method="ucb"
        ucbm="ucb4"

        fav_keys=[]
        exper = training.Experiment(searching_method = method, layers=self.layers, ucb_method=ucbm , resolution=self.resolution, bound_displacements=self.bound_displacements,  states_wasted=total_episodes, guessing_rule=self.guessing_rule, efficient_time=False)
        exper.train(bob)

        with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
            c = f.readlines()[0]
            f.close()
        fav_keys.append("run_"+str(c))

        dict["run_"+str(c)] = {}
        dict["run_"+str(c)]["label"] = ucbm
        dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs,exper.bound_displacements, exper.efficient_time,exper.ts_method]
        dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
        dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]

        plot_dict = filter_keys(dict,fav_keys)
        ploting(plot_dict, mode="stds")
        return

    def run_epgreedy1_tables(self, total_episodes=10**3):
        exper = training.Experiment(searching_method = "ep-greedy", ep=1, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="normal", min_ep=0.01, guessing_rule=self.guessing_rule, efficient_time=True, save_tables=True)
        exper.train(1)
        return

    def run_darkcounts(self,total_episodes=10**3,bobs=48):
        for method in ["ep-greedy", "ucb", "thompson-sampling"]:
            for dkr in np.arange(0,1,.05):
                exper = training.Experiment(searching_method = method, ucb_method="ucb1", ep=1, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay", time_tau=200, min_ep=0.01, guessing_rule="None", efficient_time=True, efficiency=dkr)
                exper.train(bobs)
        # return

    def run_phaseflip(self, total_episodes=10**3, bobs=48):
        for method in ["ep-greedy", "ucb", "thompson-sampling"]:
            for pf in np.arange(0,.505,.05):
                exper = training.Experiment(searching_method = method,ucb_method="ucb1", ep=1, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay", time_tau=200, min_ep=0.01, guessing_rule="None",efficient_time=True, pflip=pf)
                exper.train(bobs)
        return

    def RunAll(self, total_episodes=10**3, bob=1):
        dict={}
        method = "ep-greedy"

        fav_keys=[]
        for ep in [0.01,0.3,1]:
            exper = training.Experiment(searching_method = method, layers=self.layers, ep=ep,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="normal", min_ep=0.01, guessing_rule=self.guessing_rule, efficient_time=self.efficient_time)
            exper.train(bob)

            with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
                c = f.readlines()[0]
                f.close()

            dict["run_"+str(c)] = {}
            dict["run_"+str(c)]["label"] = str(ep) +"-greedy "
            dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs, exper.bound_displacements, exper.efficient_time,exper.ts_method]
            dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
            dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]
            fav_keys.append("run_"+str(c))

        plot_dict = filter_keys(dict,fav_keys)

        save_obj(plot_dict, "ep-greedy-Dolinar", exper.layers, exper.number_phases, exper.resolution, bob)
        ploting(plot_dict, mode="minimax")
        if bob>1:
            ploting(plot_dict, mode="stds")

        fav_keys=[]
        for tau in [200]:
            for min_ep in [0.01]:
                for method_guess in ["undefined"]:
                    exper = training.Experiment(searching_method = "ep-greedy", layers=self.layers, min_ep = min_ep, time_tau = tau,  ep=ep,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay",guessing_rule=self.guessing_rule, efficient_time=self.efficient_time, method_guess = method_guess)
                    exper.train(bob)

                    with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
                        c = f.readlines()[0]
                        f.close()

                    dict["run_"+str(c)] = {}
                    dict["run_"+str(c)]["label"] = "max("+ str(min_ep) +", e^-t/"+str(tau) +")-greedy "
                    dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs, exper.bound_displacements, exper.efficient_time,exper.ts_method]
                    dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
                    dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]

                    fav_keys.append("run_"+str(c))

        plot_dict = filter_keys(dict,fav_keys)
        save_obj(plot_dict, "exp-ep-greedy-Dolinar", exper.layers, exper.number_phases, exper.resolution, bob)
        ploting(plot_dict, mode="minimax")
        if bob>1:
            ploting(plot_dict, mode="stds")

        fav_keys=[]
        method = "ucb"
        for ucbm in ["ucb1", "ucb2", "ucb3"]:
            exper = training.Experiment(searching_method = method, layers=self.layers, ucb_method=ucbm , resolution=self.resolution, bound_displacements=self.bound_displacements,  states_wasted=total_episodes, guessing_rule=self.guessing_rule, efficient_time=self.efficient_time)
            exper.train(bob)

            with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
                c = f.readlines()[0]
                f.close()

            dict["run_"+str(c)] = {}
            dict["run_"+str(c)]["label"] = ucbm
            dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs,exper.bound_displacements, exper.efficient_time,exper.ts_method]
            dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
            dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]
            fav_keys.append("run_"+str(c))

        plot_dict = filter_keys(dict,fav_keys)
        save_obj(plot_dict, "ucbs-Dolinar", exper.layers, exper.number_phases, exper.resolution, bob, total_episodes)
        ploting(plot_dict, mode="minimax")
        if bob>1:
            ploting(plot_dict, mode="stds")

        fav_keys=[]
        method = "thompson-sampling"
        # for soft in [0.75, 1.25,1]:
        for soft in [1]:

            for mode_ts in ["None"]: #This is if you want to relate the q-table with the TS-update, but it doesnt' give any enhancement (for what i see).

                exper = training.Experiment(searching_method = method, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes, guessing_rule=self.guessing_rule, soft_ts=soft, efficient_time=self.efficient_time, ts_method=mode_ts)
                exper.train(bob)

                with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
                    c = f.readlines()[0]
                    f.close()

                dict["run_"+str(c)] = {}
                dict["run_"+str(c)]["label"] = str(soft)+"-TS"
                dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs,exper.bound_displacements, exper.efficient_time, exper.ts_method]
                dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
                dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]
                fav_keys.append("run_"+str(c))

        plot_dict = filter_keys(dict,fav_keys)
        save_obj(plot_dict, "TS", exper.layers, exper.number_phases, exper.resolution, bob)
        ploting(plot_dict, mode="minimax")
        if bob>1:
            ploting(plot_dict, mode="stds")


        save_obj(dict, "all_methods", exper.layers, exper.number_phases, exper.resolution, bob)
        return

    def RunAllDict_only(self,total_episodes=10**3, bob=1):
        dict={}
        method = "ep-greedy"
        #
        c=0
        fav_keys=[]
        for ep in [0.01,0.3,1]:

            exper = training.Experiment(searching_method = method, layers=self.layers, ep=ep,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="normal", min_ep=0.01, guessing_rule=self.guessing_rule, efficient_time=self.efficient_time)
            exper.number_bobs=bob
            c+=1

            dict["run_"+str(c)] = {}
            dict["run_"+str(c)]["label"] = str(ep) +"-greedy "
            dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs, exper.bound_displacements, exper.efficient_time,exper.ts_method]
            dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
            dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]

        for tau in [200]:
            for min_ep in [0.01]:
                for method_guess in ["undefined"]:
                    exper = training.Experiment(searching_method = "ep-greedy", layers=self.layers, min_ep = min_ep, time_tau = tau,  ep=ep,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay",guessing_rule=self.guessing_rule, efficient_time=self.efficient_time, method_guess = method_guess)
                    exper.number_bobs=bob
                    c+=1

                    dict["run_"+str(c)] = {}
                    dict["run_"+str(c)]["label"] = "max("+ str(min_ep) +", e^-t/"+str(tau) +")-greedy "
                    dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs, exper.bound_displacements, exper.efficient_time,exper.ts_method]
                    dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
                    dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]

        #
        # plot_dict = filter_keys(dict,fav_keys)
        # save_obj(plot_dict, "exp-ep-greedy-Dolinar", exper.layers, exper.number_phases, exper.resolution, bob)
        # ploting(plot_dict, mode="minimax")
        # if bob>1:
        #     ploting(plot_dict, mode="stds")

        fav_keys=[]
        method = "ucb"
        for ucbm in ["ucb1", "ucb2", "ucb3"]:
            exper = training.Experiment(searching_method = method, layers=self.layers, ucb_method=ucbm , resolution=self.resolution, bound_displacements=self.bound_displacements,  states_wasted=total_episodes, guessing_rule=self.guessing_rule, efficient_time=self.efficient_time)
            exper.number_bobs=bob
            c+=1

            dict["run_"+str(c)] = {}
            dict["run_"+str(c)]["label"] = ucbm
            dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs,exper.bound_displacements, exper.efficient_time,exper.ts_method]
            dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
            dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]
            fav_keys.append("run_"+str(c))
        # plot_dict = filter_keys(dict,fav_keys)
        # save_obj(plot_dict, "ucbs-Dolinar", exper.layers, exper.number_phases, exper.resolution, bob, total_episodes)
        # ploting(plot_dict, mode="minimax")
        # if bob>1:
        #     ploting(plot_dict, mode="stds")
        # fav_keys=[]
        method = "thompson-sampling"
        # for soft in [0.75, 1.25,1]:
        for soft in [1]:
            for mode_ts in ["None"]: #This is if you want to relate the q-table with the TS-update, but it doesnt' give any enhancement (for what i see).
                exper = training.Experiment(searching_method = method, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes, guessing_rule=self.guessing_rule, soft_ts=soft, efficient_time=self.efficient_time, ts_method=mode_ts)
                exper.number_bobs=bob
                c+=1
                dict["run_"+str(c)] = {}
                dict["run_"+str(c)]["label"] = str(soft)+"-TS"
                dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs,exper.bound_displacements, exper.efficient_time, exper.ts_method]
                dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
                dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]
                fav_keys.append("run_"+str(c))
        # plot_dict = filter_keys(dict,fav_keys)
        # save_obj(plot_dict, "TS", exper.layers, exper.number_phases, exper.resolution, bob)
        # ploting(plot_dict, mode="minimax")
        # if bob>1:
        #     ploting(plot_dict, mode="stds")
        print(bob)
        print(dict)
        save_obj(dict, "all_methods", exper.layers, exper.number_phases, exper.resolution, bob)
        return
### BANDIT ###
# mega = MegaFrontEnd(layers=1, guessing_rule="Dolinar", bound_displacements=0.7, resolution=0.7, efficient_time = False)
# mega.RunAll(total_episodes=1000, bob=500)
# mega.RunInteresting(total_episodes=10**2, bob=1)
#
#### 2 LAYERS ###
# mega = MegaFrontEnd(layers=2, guessing_rule="None")
# # mega.RunAll(total_episodes=10**5, bob=48)
# mega.RunAllDict_only(total_episodes=10**5, bob=48)

# mega.RunUCB_TS(total_episodes=5*10**5, bob=48)
# mega.single_run()
if __name__ == "__main__":
    mega = MegaFrontEnd(layers=2, guessing_rule="None")
    mega.run_epgreedy1_tables(total_episodes=10**8)
