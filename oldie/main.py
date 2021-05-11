import training
import gc
from showing import ploting
import pickle
from misc import save_obj, filter_keys
import numpy as np
import os
import matplotlib
from training import Experiment
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

class MegaFrontEnd():
    def __init__(self, layers=2, resolution=0.1, bound_displacements=1, guessing_rule="None", efficient_time = False):
        self.layers = layers
        self.guessing_rule = guessing_rule
        self.resolution = resolution
        self.bound_displacements = bound_displacements
        self.efficient_time = efficient_time

    def sweep_energies(self,total_episodes,bobs):
        for amp in np.arange(.1,1.6,.1):
            for method in ["ep-greedy", "ucb", "thompson-sampling"]:
                exper = training.Experiment(amplitude=amp, searching_method = method, layers=self.layers, ucb_method="ucb1" , resolution=self.resolution, bound_displacements=self.bound_displacements,  states_wasted=total_episodes, guessing_rule="None", efficient_time=True)
                exper.train(bobs)
                del exper
        return

    def ep_greedy_tables(self, total_episodes=10**3, ep=0.3, bob=1):
        exper = training.Experiment(searching_method = "ep-greedy", ep=ep, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="normal", min_ep=0.01, guessing_rule=self.guessing_rule, efficient_time=True, save_tables=True)
        exper.train(bob)
        return

    def ucb_tables(self, total_episodes=5*10**5, bob=1):
        exper = training.Experiment(searching_method = "ucb",ucb_method="ucb1", ep=1, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="normal", min_ep=0.01, guessing_rule=self.guessing_rule, efficient_time=True, save_tables=True)
        exper.train(bob)
        return

    def run_darkcounts(self,total_episodes=10**3,bobs=48):
        for method in ["ep-greedy", "ucb", "thompson-sampling"]:
            for dkr in np.arange(.1,1,.2):
                exper = training.Experiment(searching_method = method, ucb_method="ucb1", ep=1, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay", time_tau=200, min_ep=0.01, guessing_rule="None", efficient_time=True, efficiency=dkr)
                exper.train(bobs)
        # return

    def run_phaseflip(self, total_episodes=10**3, bobs=48):
        for method in ["ep-greedy", "ucb", "thompson-sampling"]:
            for pf in np.arange(0.05,.505,.1):
                exper = training.Experiment(searching_method = method,ucb_method="ucb1", ep=1, layers=self.layers,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay", time_tau=200, min_ep=0.01, guessing_rule="None",efficient_time=True, pflip=pf)
                exper.train(bobs)
        return


    def ucbs(self, total_episodes, bob=12):
        dict = {}
        method="ucb"
        fav_keys=[]
        for ucbm in ["ucb1","ucb2","ucb3"]:
            exper = training.Experiment(searching_method = method, layers=self.layers, ucb_method=ucbm , resolution=self.resolution, bound_displacements=self.bound_displacements,  states_wasted=total_episodes, guessing_rule=self.guessing_rule, efficient_time=False)
            exper.train(bob)
            with open(str(exper.layers)+"L"+str(exper.number_phases)+"PH"+str(exper.resolution)+"R/number_rune.txt", "r") as f:
                c = f.readlines()[0]
                f.close()
            fav_keys.append("run_"+str(c))
            dict["run_"+str(c)] = {}
            dict["run_"+str(c)]["label"] = ucbm.capitalize()
            dict["run_"+str(c)]["info"] = [exper.number_phases, exper.amplitude, exper.layers, exper.resolution, exper.searching_method, exper.guessing_rule, exper.method_guess, exper.number_bobs,exper.bound_displacements, exper.efficient_time,exper.ts_method]
            dict["run_"+str(c)]["info_ep"] = [exper.ep_method, exper.ep, exper.min_ep, exper.time_tau]
            dict["run_"+str(c)]["info_ucb"] = [exper.ucb_method]
        plot_dict = filter_keys(dict,fav_keys)
        ploting(plot_dict, mode="stds")
        return

    def EnhancedQL(self, total_episodes=10**3, bob=1, plots=False):
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
        save_obj(plot_dict, "ep-greedy", exper.layers, exper.number_phases, exper.resolution, bob)
        fav_keys=[]
        for tau in [200]:
            for min_ep in [0.01]:
                exper = training.Experiment(searching_method = "ep-greedy", layers=self.layers, min_ep = min_ep, time_tau = tau,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay",guessing_rule=self.guessing_rule, efficient_time=self.efficient_time)
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
        save_obj(plot_dict, "exp-ep-greedy", exper.layers, exper.number_phases, exper.resolution, bob)
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
        save_obj(plot_dict, "ucbs", exper.layers, exper.number_phases, exper.resolution, bob, total_episodes)
        fav_keys=[]
        method = "thompson-sampling"
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

        method = "ep-greedy"
        ep="TS+0.01exp"
        fav_keys=[]
        exper = training.Experiment(searching_method = "ep-greedy", layers=self.layers, min_ep = 0.01, time_tau = 200,  ep=0.01,resolution=self.resolution, bound_displacements=self.bound_displacements, states_wasted=total_episodes,ep_method="exp-decay",guessing_rule="None", efficient_time=self.efficient_time, method_guess = "thompson-sampling")
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
        save_obj(plot_dict, "ep-TS", exper.layers, exper.number_phases, exper.resolution, bob)
###################################################################################################3###
        save_obj(dict, "all_methods", exper.layers, exper.number_phases, exper.resolution, bob)
        if plots == True:

            mode_log="on"

            # os.system("python3 trad_ql.py")
            # os.system("python3 enhanced_ql.py")
            # os.system("python3 trad_ql.py")

            matplotlib.rc('font', serif='cm10')
            matplotlib.rc('text', usetex=True)
            plt.rcParams.update({'font.size': 45})

            color1="purple"
            color2 = (225/255, 15/255, 245/255)
            color3 = (150/255, 22/255, 9/255)
            color_2l = [46/255, 30/255, 251/255]
            colorexp = (13/255,95/255,14/255)

            colorucb1 = (19/255, 115/255,16/255)
            colorucb2 = (170/255,150/255,223/255)
            colorucb3 = (74/255, 90/255, 93/255)
            colors = {"run_1": "orange", "run_2": color2, "run_3":"brown", "run_4":colorexp, "run_5":colorucb1, "run_6": colorucb2, "run_7":colorucb3, "run_8":"yellow", "run_9":"purple"}

            labels = {"run_1":r'$\epsilon = 0.01$'+"-greedy" , "run_2": r'$\epsilon = 0.3$'+"-greedy", "run_3": r'$\epsilon = 1$'+"-greedy", "run_4":"max(0.01, "+r'$e^{-t/\tau}$'+")-greedy", "run_5":"UCB-1","run_6":"UCB-2","run_7":"UCB-3","run_8":"TS", "run_9":"max(0.01, "+r'$e^{-t/\tau}$'+")-greedy + TS"} #


            ####### Q-LEARNING PLOT ######
            ####### Q-LEARNING PLOT ######
            ####### Q-LEARNING PLOT ######
            ####### Q-LEARNING PLOT ######

            interesting = ["run_1","run_2", "run_3","run_4"]
            dict_plot = {}
            print(dict.keys())
            for i in interesting:
                dict_plot[i] = dict[i]
            for run in interesting:
                dict_plot[run]["label"] = labels[run]

            plt.figure(figsize=(30,22), dpi=150)
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
            axinticks=[]

            name = str(dict.keys())

            once=True
            neg="Not"

            print("ploting Q learning")
            for run in dict_plot.keys():

                number_phases, amplitude, layers, resolution, searching_method, guessing_rule, method_guess, number_bobs, bound_displacements,efficient_time, ts_method = dict[run]["info"]
                exp = Experiment(number_phases=number_phases, amplitude= amplitude, layers=layers, resolution=resolution, bound_displacements=bound_displacements)
                exp.load_data(run)
                run_color = colors[run]
                if mode_log == "on":
                    times = np.log10(exp.results[0])
                else:
                    times = exp.results[0]
                if once==True:
                    ax1.plot(times,exp.optimal_value*np.ones(len(exp.results[0])), '--',linewidth=9, alpha=0.8, label=r'$P_*^{(2)}$', color=color_2l)
                    ax1.plot([times[0], times[-1]], [exp.homodyne_limit]*2, '--', linewidth=9 , color="black", label="Homodyne limit")

                    ax2.plot([times[0], times[-1]], [exp.homodyne_limit]*2, '--', linewidth=9 , color="black", label="Homodyne limit")
                    ax2.plot(times,exp.optimal_value*np.ones(len(times)), '--',linewidth=9, alpha=0.6, color=color_2l)

                    axins = zoomed_inset_axes(ax2, zoom=2.7,loc="lower right")
                    loc1=-int(len(exp.results[0])*0.7)
                    loc2=-1
                    once=False
                ax1.plot(times, exp.results[1]/exp.results[0], linewidth=9, alpha=0.9,label=dict[run]["label"], color=run_color)
                ax2.plot(times, exp.results[2],linewidth=3 ,alpha=.5, label=dict[run]["label"], color=run_color)
                axins.plot(np.log10(exp.results[0][loc1:loc2]), np.log10(exp.results[2][loc1:loc2]), '-', linewidth=9,alpha=.8, color=colors[run], label=dict[run]["label"])
                axins.plot(np.log10([exp.results[0][loc1], exp.results[0][loc2-1]]), np.log10([1-exp.opt_2l]*2), '-.', alpha=.8, linewidth=9,color=color_2l,
                label="Optimal 2L")
                axinticks.append(np.log10(exp.results[2][loc1]))
                ax1.fill_between(times, (exp.results[1] - exp.stds[0]/2)/exp.results[0],
                (exp.results[1] + exp.stds[0]/2)/exp.results[0], alpha=.4, color=run_color)
                ax2.fill_between(times,  np.log10(exp.results[2] - exp.stds[1]/2) , np.log10(exp.results[2] + exp.stds[1]/2), alpha=0.4, color=run_color)
                ax1.legend()
                mark_inset(ax2, axins, loc1=1, loc2=2, fc="green", ec="0.3", alpha=0.5)
                axinticks.append(exp.opt_2l)
                yticks = np.arange(np.round(min(exp.results[2]),3),1-exp.opt_2l,.1)
                ax1.set_yticks(yticks)
                ax2.set_yticks(yticks)
                axins.set_yticks(axinticks)
                axins.set_yticklabels([str(np.round(i,3)) for i in axinticks])
                plt.setp(axins.get_yticklabels(), size=27)
                plt.setp(axins.get_xticklabels(), visible=False)
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax2.set_xticks([0,1,2,3,4,5,np.log10(5*10**5)])
                ax2.set_xticklabels([r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$',r'$10^{4}$',r'$10^{5}$',r'$5 \; 10^{5}$'])
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
                ax2.tick_params(axis='x', which='both',top='off')
                ax1.legend(loc="lower right", prop={"size":35})
                ax2.set_xlabel("t", size=54)
                ax1.set_ylabel(r'\textbf{R}$_t$', size=54)
                ax2.set_ylabel(r'\textbf{P}$_t$', size=54)

            inf = dict[run]["info"]
            layers, phases, resolution = inf[2], inf[0], inf[3]
            plt.savefig(str(layers) + "L" + str(phases) + "PH"+str(resolution) + "R/figures/Qlearning.png")



                ####### ENH-Q-LEARINIG PLOT ######
                ####### ENH-Q-LEARINIG PLOT ######
                ####### ENH-Q-LEARINIG PLOT ######
                ####### ENH-Q-LEARINIG PLOT ######


            interesting = ["run_2","run_9", "run_8","run_5"]
            dict_plot = {}
            print(dict.keys())
            for i in interesting:
                dict_plot[i] = dict[i]
            for run in interesting:
                dict_plot[run]["label"] = labels[run]

            plt.figure(figsize=(30,22), dpi=150)
            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
            axinticks=[]

            name = str(dict.keys())

            once=True
            neg="Not"

            for run in dict_plot.keys():
                print(run)
                number_phases, amplitude, layers, resolution, searching_method, guessing_rule, method_guess, number_bobs, bound_displacements,efficient_time, ts_method = dict[run]["info"]
                exp = Experiment(number_phases=number_phases, amplitude= amplitude, layers=layers, resolution=resolution, bound_displacements=bound_displacements)
                exp.load_data(run)
                run_color = colors[run]
                if mode_log == "on":
                    times = np.log10(exp.results[0])
                else:
                    times = exp.results[0]
                if once==True:
                    ax1.plot(times,exp.optimal_value*np.ones(len(exp.results[0])), '--',linewidth=9, alpha=0.8, label=r'$P_*^{(2)}$', color=color_2l)
                    ax1.plot([times[0], times[-1]], [exp.homodyne_limit]*2, '--', linewidth=9 , color="black", label="Homodyne limit")

                    ax2.plot([times[0], times[-1]], [exp.homodyne_limit]*2, '--', linewidth=9 , color="black", label="Homodyne limit")
                    ax2.plot(times,exp.optimal_value*np.ones(len(times)), '--',linewidth=9, alpha=0.6, color=color_2l)

                    axins = zoomed_inset_axes(ax2, zoom=2.7,loc="lower right")
                    loc1=-int(len(exp.results[0])*0.35)
                    loc2=-1
                    once=False
                ax1.plot(times, exp.results[1]/exp.results[0], linewidth=9, alpha=0.9,label=dict[run]["label"], color=run_color)
                ax2.plot(times, exp.results[2],linewidth=3 ,alpha=.5, label=dict[run]["label"], color=run_color)
                axins.plot(np.log10(exp.results[0][loc1:loc2]), np.log10(exp.results[2][loc1:loc2]), '-', linewidth=9,alpha=.8, color=colors[run], label=dict[run]["label"])
                axins.plot(np.log10([exp.results[0][loc1], exp.results[0][loc2-1]]), np.log10([1-exp.opt_2l]*2), '-.', alpha=.8, linewidth=9,color=color_2l,
                label="Optimal 2L")
                axinticks.append(np.log10(exp.results[2][loc1]))
                ax1.fill_between(times, (exp.results[1] - exp.stds[0]/2)/exp.results[0],
                (exp.results[1] + exp.stds[0]/2)/exp.results[0], alpha=.4, color=run_color)
                ax2.fill_between(times,  np.log10(exp.results[2] - exp.stds[1]/2) , np.log10(exp.results[2] + exp.stds[1]/2), alpha=0.4, color=run_color)
                ax1.legend()
                mark_inset(ax2, axins, loc1=1, loc2=2, fc="green", ec="0.3", alpha=0.5)
                axinticks.append(exp.opt_2l)
                yticks = np.arange(np.round(min(exp.results[2]),3),1-exp.opt_2l,.1)
                ax1.set_yticks(yticks)
                ax2.set_yticks(yticks)
                axins.set_yticks(axinticks)
                axins.set_yticklabels([str(np.round(i,3)) for i in axinticks])
                plt.setp(axins.get_yticklabels(), size=27)
                plt.setp(axins.get_xticklabels(), visible=False)
                plt.setp(ax1.get_xticklabels(), visible=False)
                ax2.set_xticks([0,1,2,3,4,5,np.log10(5*10**5)])
                ax2.set_xticklabels([r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$',r'$10^{4}$',r'$10^{5}$',r'$5 \; 10^{5}$'])
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.1)
                ax2.tick_params(axis='x', which='both',top='off')
                ax1.legend(loc="lower right", prop={"size":35})
                ax2.set_xlabel("t", size=54)
                ax1.set_ylabel(r'\textbf{R}$_t$', size=54)
                ax2.set_ylabel(r'\textbf{P}$_t$', size=54)

            inf = dict[run]["info"]
            layers, phases, resolution = inf[2], inf[0], inf[3]
            plt.savefig(str(layers) + "L" + str(phases) + "PH"+str(resolution) + "R/figures/ENH-QL.png")


        return

if __name__ == "__main__":
    mega = MegaFrontEnd(layers=2, guessing_rule="None")

    mega.EnhancedQL(total_episodes=10**2, bob=2, plots=True)
    # mega.ucb_tables(total_episodes=5*10**5, bob=24)
    # mega.sweep_energies(total_episodes=5*10**5, bob=24)
    # mega.run_phaseflip()
    # mega.run_darkcounts()
