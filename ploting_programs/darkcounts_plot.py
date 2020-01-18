import numpy as np
import matplotlib.pyplot as plt
from training import Experiment
import os
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from datetime import datetime
import pickle
import matplotlib

# matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
matplotlib.rc('font', serif='cm10')
matplotlib.rc('text', usetex=True)

axinticks=[]
plt.rcParams.update({'font.size': 100})

plt.figure(figsize=(50,30), dpi=80)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.1)

ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))
colors={"0":"red","1":"blue","2":"green"}
labels={"0":"UCB-1", "1":"max(0.01,"+r'$e^{-\frac{t}{\tau}}$'+")-greedy", "2":"TS"}
darkcounts = np.loadtxt("DKrs.csv")
darkcounts=np.array(darkcounts)
for a in [ax1,ax2]:
    a.set_xticks(np.arange(0,1.1,.2))
    for tick in a.xaxis.get_major_ticks():
                tick.label.set_fontsize(40)
                # tick.label.set_rotation('vertical')
eff=0
ax1.plot(np.linspace(0,1,len(darkcounts)),1-darkcounts,'--', alpha=0.6, linewidth=8, color="black", label=r'$p_*^{(L=2)}$')
ax2.plot(np.linspace(0,1,len(darkcounts)),1-darkcounts,'--', alpha=0.6, linewidth=8, color="black", label=r'$p_*^{(L=2)}$')
#

size=2000
for r in range(3,37):
    # exp = Experiment(number_phases=2, amplitude= .4, layers=2, resolution=.1, bound_displacements=1)
    # exp.load_data("run_"+str(r))
    lc = np.load("2L2PH0.1R/run_"+str(r)+"/learning_curves_data.npy", allow_pickle=True)
    cumrefin = lc[1][-1]/lc[0][-1]
    prosucgre = lc[3][-1]
    if r<6:
        ax1.scatter(eff, cumrefin, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
        ax2.scatter(eff, prosucgre, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])

    else:
        ax1.scatter(eff, cumrefin, s=size, alpha=0.7,color=colors[str(r%3)])
        ax2.scatter(eff, prosucgre, s=size, alpha=0.7,color=colors[str(r%3)])

    if (r%3==0)&(r!=3):
        eff+=0.1

ax1.legend(loc="upper left", prop={"size":70})
ax2.legend(loc="upper left", prop={"size":70})
ax1.set_xlabel(r'$1-p_{dc}$', size=100, labelpad=15)
ax2.set_xlabel(r'$1-p_{dc}$', size=100, labelpad=15)
ax2.yaxis.set_label_position("right")

ax1.set_ylabel(r'\textbf{R}$_t$', size=170)
ax2.set_ylabel(r'\textbf{P}$_t$', size=170, labelpad=40)
plt.savefig("darkcounts.pdf")
# plt.show()
