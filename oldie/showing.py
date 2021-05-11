import numpy as np
import matplotlib.pyplot as plt
from training import Experiment
import os
from datetime import datetime
import pickle



def ploting(dict, mode="minimax", mode_log="on", save=True, show=False, particular_name="std"):
    """ Function that plots what's inside of the dictionary, obtained in the corresponding front_end program.

        mode: when averaging many learners, choice between ploting, at each time, the minimum value among all agents and the maximum (minimax), or the standard deviation (stds).

        mode_log: put log10(x) if on, else not
    """


    #details = [energy, layers, resolution]
    #dict is a dictornary with the labels you want to assign
    plt.figure(figsize=(16,10)  , dpi=70)
    ax1 = plt.subplot2grid((2,1), (0,0))
    ax2 = plt.subplot2grid((2,1), (1,0))
    name = str(dict.keys())

    once=True
    neg="Not"
    for run in dict.keys():
        number_phases, amplitude, layers, resolution, searching_method, guessing_rule, method_guess, number_bobs, bound_displacements,efficient_time, ts_method = dict[run]["info"]
        exp = Experiment(number_phases=number_phases, amplitude= amplitude, layers=layers, resolution=resolution, bound_displacements=bound_displacements)
        exp.load_data(run)
        run_color = tuple(np.random.randint(256, size=3)/256)


        if mode_log == "on":
            times = np.log10(exp.results[0])
        else:
            times = exp.results[0]

        if efficient_time == "True":

            ax1.plot(times, exp.results[1]/exp.results[0], '.', alpha=0.5, color=run_color)
            ax2.plot(times, exp.results[2], '.', color=run_color)

        ax1.plot(times, exp.results[1]/exp.results[0], linewidth=4, alpha=0.8,label=dict[run]["label"], color=run_color)
        ax2.plot(times, exp.results[2],linewidth=4 ,alpha=0.8, label=dict[run]["label"], color=run_color)

        if once==True:
            ax1.plot(times,exp.optimal_value*np.ones(len(exp.results[0])))
            ax2.plot(times,exp.optimal_value*np.ones(len(times)))
            once=False

        if number_bobs>1:
            if mode == "minimax":

                ax1.plot(times, exp.minimax[0]/exp.results[0], '--',linewidth=2,  color=run_color)
                ax1.plot(times, exp.minimax[1]/exp.results[0], '--',linewidth=2,  color=run_color)

                ax2.plot(times, exp.minimax[2], '--',linewidth=2,  color=run_color)
                ax2.plot(times, exp.minimax[3], '--',linewidth=2,  color=run_color)


                ax1.fill_between(times, exp.minimax[0]/exp.results[0],
                 exp.minimax[1]/exp.results[0], alpha=.4, color=run_color)
                ax2.fill_between(times, exp.minimax[2], exp.minimax[3], alpha=.2, color=run_color)

            elif mode == "stds":
                # print(exp.stds[2])
                ax1.plot(times, (exp.results[1] - exp.stds[0]/2)/exp.results[0], '--',linewidth=2,  color=run_color)
                ax1.plot(times, (exp.results[1] + exp.stds[0]/2)/exp.results[0], '--',linewidth=2,  color=run_color)

                ax2.plot(times, exp.results[2] - exp.stds[1]/2, '--',linewidth=2,  color=run_color)


                ax1.fill_between(times, (exp.results[1] - exp.stds[0]/2)/exp.results[0],
                (exp.results[1] + exp.stds[0]/2)/exp.results[0], alpha=.4, color=run_color)
                ax2.fill_between(times,  exp.results[2] - exp.stds[1]/2 ,  exp.results[2] + exp.stds[1]/2, alpha=.2, color=run_color)
            else:
                print("specify a method to show the deviation: minimax or stds")
        del exp
    ax1.legend()
    ax2.legend()
    if mode_log=="off":
        name = "lx_off"+name
    if save == True:
        inf = dict[run]["info"]
        layers, phases, resolution = inf[2], inf[0], inf[3]
        if particular_name != "std":
            plt.savefig(str(layers) + "L" + str(phases) + "PH"+str(resolution) + "R/figures/"+particular_name+str(mode))
        else:
            plt.savefig(str(layers) + "L" + str(phases) + "PH"+str(resolution) + "R/figures/"+name+"-"+str(mode))
    if show == True:
        plt.show()


if __name__ == "__main__":

    from misc import load_obj
    # dict = load_obj("all_favourite_methods_x1000", resolution=0.7)
    # dict = load_obj("exp-ep-greedy-Dolinar_x500", resolution=0.7)
    name = "all_methods_x12_ep100"
    dict = load_obj(name, resolution=0.1, layers=2)

    # dict = load_obj("ep-greedy-Dolinar_x1", resolution=0.33)
    # # for i in dict.keys():
    # #     print(i, dict[i]["label"])
    # interesting = ["run_10","run_16", "run_2"]
    # interesting = ["run_1","run_2", "run_3", "run_4","run_5"]
    interesting = ["run_1","run_2", "run_3"]
    # interesting = dict.keys()
    # #
    dict_plot = {}
    for i in interesting:
        dict_plot[i] = dict[i]
    ploting(dict_plot,mode_log="off",save=True,show=True, particular_name=name,mode="stds")
