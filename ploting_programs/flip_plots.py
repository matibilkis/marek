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
plt.rcParams.update({'font.size': 80})

plt.figure(figsize=(50,30), dpi=80)
# plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.1)

ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))
colors={"0":"red","1":"blue","2":"green"}
labels={"1":"UCB-1", "2":"max(0.01,"+r'$e^{-\frac{t}{\tau}}$'+")-greedy", "0":"TS"}
flips = np.loadtxt("PFLIP.csv")
flips=np.array(flips)
for k in range(12):
    flips = np.delete(flips,-1)


                # tick.label.set_rotation('vertical')
eff=0
ax1.plot(np.linspace(0,flips[-1],len(flips)),1-flips,'--', alpha=0.6, linewidth=8, color="black", label=r'$p_*^{(L=2)}$')
ax2.plot(np.linspace(0,flips[-1],len(flips)),1-flips,'--', alpha=0.6, linewidth=8, color="black", label=r'$p_*^{(L=2)}$')
#

size=2000
for r in range(1,21):
    lc = np.load("2L2PH0.1R/run_"+str(r)+"/learning_curves_data.npy", allow_pickle=True)
    cumrefin = lc[1][-1]/lc[0][-1]
    prosucgre = lc[3][-1]
    if r<4:
        ax1.scatter(eff, cumrefin, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
        ax2.scatter(eff, prosucgre, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
    else:
        ax1.scatter(eff, cumrefin, s=size, alpha=0.7,color=colors[str(r%3)])
        ax2.scatter(eff, prosucgre, s=size, alpha=0.7,color=colors[str(r%3)])
    if (r%3==0):
        eff+=0.05
for r in range(24,34):
    lc = np.load("2L2PH0.1R/run_"+str(r)+"/learning_curves_data.npy", allow_pickle=True)
    cumrefin = lc[1][-1]/lc[0][-1]
    prosucgre = lc[3][-1]
    if r<4:
        ax1.scatter(eff, cumrefin, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
        ax2.scatter(eff, prosucgre, s=size, color=colors[str(r%3)],alpha=0.7, label=labels[str(r%3)])
    else:
        ax1.scatter(eff, cumrefin, s=size, alpha=0.7,color=colors[str(r%3)])
        ax2.scatter(eff, prosucgre, s=size, alpha=0.7,color=colors[str(r%3)])
    if (r%3==0):
        eff+=0.05

for a in [ax1,ax2]:
    a.set_xticks(np.arange(0,eff+0.05,.1))
    for tick in a.xaxis.get_major_ticks():
                tick.label.set_fontsize(40)
ax1.legend(loc="lower left", prop={"size":60})
ax2.legend(loc="lower left", prop={"size":60})
ax1.set_xlabel(r'$p_f$', size=100, labelpad=25)
ax2.set_xlabel(r'$p_f$', size=100, labelpad=25)
ax2.yaxis.set_label_position("right")

ax1.set_ylabel(r'\textbf{R}$_t$', size=170)
ax2.set_ylabel(r'\textbf{P}$_t$', size=170, labelpad=35)
plt.savefig("flips.pdf")
# plt.show()
