import os


import numpy as np
import matplotlib.pyplot as plt
from front_end import Discriminate
#
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

plt.rcParams.update({'font.size': 45})
# plt.rc('text', usetex=True)
# plt.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]
# colors = {"ucb": tuple([0.82, .12, .35, .07]), 'thompson-sampling': tuple([0.90, .94, .01, .03]), 'ep-greedy': tuple([0.12, .87, .95, .33])}
#
markers={"thompson-sampling":"*", "ucb":"v", "ep-greedy":"o"}
color1 = (69/255, 209/255, 154/255)
color2 = (225/255, 15/255, 245/255)
color3 = (150/255, 22/255, 9/255)
color_2l = [46/255, 30/255, 251/255]

def color(number):
    np.random.seed(number)
    colors[str(number)] = (np.random.random(), np.random.random(), np.random.random())
    return
# colors = {"ucb": color1, 'thompson-sampling': color2, 'ep-greedy': color3}
colors = {"0": color1, '1': color2, '2': color3}
for i in range(3,10):
    color(i)

labels= {"ucb": "UCB", 'thompson-sampling': "Thompson Sampling", 'ep-greedy': r'$\epsilon$-greedy'}
plt.figure(figsize=(30,30), dpi=35)

# brute_here = np.load("minima_with_model/brute-force-brute/0.56/l2ph2r0.1b1.npy",allow_pickle=True)[1]


axinticks=[]
labels_all = {"run_1":r'$\epsilon = 0.01$'+"-greedy" , "run_2": r'$\epsilon = Max(0.01, e^{-t/\tau}$)', "run_3": "1-greedy", "run_4":"ucb ucb", "run_5": "exp-"+r'$\epsilon$' + " + TS", "run_6": "exp-gre + 0.1-TS", "run_7": "TS-TS", "run_8": "0.1(TS+TS)", "run_9": "ucb + ucb (banditalg)", "run_10": "ucb+ucb (anormal)", "run_11": "ucb + TS", "run_12": "exp-"+r'$\epsilon$' + " +ucb"}


labels = {"run_1":r'$\epsilon = 0.01$'+"-greedy" , "run_2": r'$\epsilon = Max(0.01, e^{-t/1000}$)', "run_3": "1-greedy", "run_4":"UCB-1", "run_5": "exp-"+r'$\epsilon$' + " + TS", "run_6": "exp-gre + 0.1-TS", "run_7": "TS", "run_8": "0.1(TS+TS)", "run_9": "UCB-2", "run_10": "UCB-3)", "run_11": "ucb + TS", "run_12": "exp-"+r'$\epsilon$' + " +ucb", "run_55": "0.3-greedy", "run_56": r'$\epsilon = Max(0.27, e^{-t/5000})$'}

labels = { 'run_1' : '1-greedy',  'run_2' : 'exp-greedy',  'run_3' : '0.3-greedy',  'run_4' : 'UCB-1',  'run_5' : 'exp-greedy + TS',  'run_6' : 'TS',  'run_7' : '0.1-TS',  'run_8' : 'UCB-2',  'run_9' : 'UCB-3' }


runnings = {}
# runnings["ql"] = ["run_2","run_3", "run_55", "run_56"] #Q-learning plot
runnings["ql"] = ["run_2","run_3", "run_1"] #Q-learning plot

# runnings["comp_ep"] = ["run_2", "run_55", "run_56"]
# runnings["enhql"] = ["run_2","run_4", "run_5" ,"run_7"] #Enhanced q-learning plot
runnings["ucbs_methods"] = ["run_4","run_8", "run_9"] #comparing ucb methods plot
# runnings["exp_enh_only_guess"] = ["run_2", "run_5", "run_12", "run_6"] #enhaced exp-epsilon greedy with the guess only (ucb or TS)
runnings["ts"] = ["run_7", "run_6"]

insets={"ql":True, "enhql":True, "ucbs_methods": True, "exp_enh_only_guess":True, "ts":True}


# for ress in [0.1, 0.01]:
for ress in [0.1]:

    # for keys in runnings.keys():
    for keys in ["ts"]:
        rruns = runnings[keys]
        inset = False
        first=True
        jj=0
        for runs in rruns:
            re = Discriminate(layers=2, number_phases=2, resolution=0.1)
            re.grab_data(runs)

            for i in range(len(re.results)):
                re.results[i] = re.results[i]

            with open(str(runs)+"/info.txt", 'r') as f:
                for number,line in enumerate(f):
                    if number == 2:
                        method=line[:-1].replace(' search_method: ','')


            if first == True:

                ax2 = plt.subplot2grid((2,1), (1,0))
                ax1 = plt.subplot2grid((2,1), (0,0), sharex=ax2)
                if inset:
                    axins = zoomed_inset_axes(ax2, zoom=3,loc="lower right")

                first=False
            re.results[0] = np.array(re.results[0]) +1

            ax1.plot(np.log10(re.results[0]), re.results[1]/np.array(re.results[0]),markersize=17,  linewidth=5,alpha=.7, color=colors[str(jj)], label=labels[str(runs)])
            ax1.set_ylabel(r'$R_t$', size=54)
            ax1.fill_between(np.log10(re.results[0]), (re.results[1] - re.results[2]/2)/np.array(re.results[0]),(re.results[1] + re.results[2]/2)/np.array(re.results[0]),alpha=.6, facecolor=colors[str(jj)])


            loc1=-int(len(re.results[0])*0.2)
            loc2=-1

            ax2.plot(np.log10(re.results[0]), np.log10(re.results[3]), '-', linewidth=9,alpha=.7, color=colors[str(jj)], label=str(method))
            ax2.set_ylabel(r'$P_t$')
            ax2.fill_between(np.log10(re.results[0]), np.log10((re.results[3] - re.results[-1]/2)),np.log10((re.results[3] + re.results[-1]/2)),alpha=.6, facecolor=colors[str(jj)])

            if inset:
                axins.plot(np.log10(re.results[0][loc1:loc2]), np.log10(re.results[3][loc1:loc2]), '-', linewidth=9,alpha=.9, color=colors[str(jj)], label=str(method))

                axins.fill_between(np.log10(re.results[0][loc1:loc2]), np.log10((re.results[3][loc1:loc2] - re.results[-1][loc1:loc2]/2)),np.log10((re.results[3][loc1:loc2] + re.results[-1][loc1:loc2]/2)),alpha=.2, facecolor=colors[str(jj)])

                axinticks.append(re.results[3][loc1])
                axins.plot(np.log10([re.results[0][loc1], re.results[0][loc2-1]]), np.log10([re.exp.opt_2l]*2), '-.', alpha=.8, linewidth=9,color=color_2l,
                label="Optimal 2L")


            os.chdir("..")
            jj+=1
        ax2.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), np.log10([re.exp.env_example.helstrom()]*2), '--', linewidth=9, alpha=.9, color="black", label="Helstrom")
        ax2.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), np.log10([re.exp.agent_example.homodyne()]*2), '--', linewidth=9 , color="green", label="Homodyne")
        # ax2.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), np.log10([0.8934268313801327]*2), '--', linewidth=9 , color="purple", label="Optimal Kennedy")
        ax2.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), np.log10([re.exp.opt_2l]*2), '--', alpha=.8, linewidth=9,color=color_2l,
        label="Optimal 2L")

        ax1.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), [re.exp.env_example.helstrom()]*2, '--', linewidth=9,alpha=.9, color="black",
        label="Helstrom")
        ax1.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), [re.exp.opt_2l]*2, '--', alpha=.8, linewidth=9,color=color_2l,
        label="Optimal 2L")
        ax1.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), [re.exp.agent_example.homodyne()]*2, '--', linewidth=9 , color="green", label="Homodyne")
        # ax1.plot(np.log10([np.min(re.results[0]), np.max(re.results[0])]), [re.opt_kenn]*2, '--', linewidth=9 , color="purple", label="Optimal Kennedy")
        ax2.tick_params(axis='x', which='both',top='off')

        # ax1.legend(loc="lower right", prop={"size":20})
        ax1.legend(loc="lower right", prop={"size":30})
        ax2.set_xlabel("t", size=54)
        # ax1.legend(bbox_to_anchor=(0.55,0.35),prop={"size":35}, bbox_transform=plt.gcf().transFigure)


        if inset:
            mark_inset(ax2, axins, loc1=1, loc2=2, fc="green", ec="0.3", alpha=0.5)
            axinticks.append(re.exp.opt_2l)
            plt.setp(axins.get_xticklabels(), visible=False)
            axins.set_yticks(np.log10(axinticks))
            axins.set_yticklabels([str(np.round(i,3)) for i in axinticks])
            plt.setp(axins.get_yticklabels(), size=27)


        yticks = np.linspace(.4,.93,8)
        ax2.set_yticks(np.log10(yticks))
        ax2.set_yticklabels([str(np.round(i,2)) for i in yticks])

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.set_xticklabels(['',r'$10^{0}$',r'$10^{1}$',r'$10^{2}$',r'$10^{3}$',r'$10^{4}$',r'$10^{5}$',r'$5 10^{5}$',''])
        plt.savefig("plots/ql_final_paper_ts.pdf")
        # plt.show()


        # plt.savefig("ql_various_no_caption.pdf",dpi=300)
        # plt.savefig("figures_res" + str(ress)+"/"+keys + ".pdf")
