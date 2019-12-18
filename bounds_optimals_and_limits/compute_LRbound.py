from basics import Basics
import numpy as np
import matplotlib.pyplot as plt
from misc import Kull

b = Basics(layers=1,bound_displacements=.7, resolution=.7)
b.define_actions()
probs=[]
for betas in b.actions:
    for outcomes in [0,1]:
        probs.append(b.P(np.sign(betas+0.00001)*(-1)**(outcomes+1)*b.amplitude, betas, 1, outcomes))

print(probs)
print((probs[0] + probs[1] )/2, b.err_dolinar(b.actions[0]))
print((probs[2]+probs[3])/2, b.err_dolinar(b.actions[1]))

optimal = max(probs)
reg_coeff=0
for p in probs:
    if p!=optimal:
        reg_coeff+=(optimal-p)/Kull(p,optimal)
# dol_prob= [b.err_dolinar(disp) for disp in b.actions]
# optimal = max(dol_prob)
# pos_optimal = np.where(dol_prob == optimal)[0][0]
#
# #
# reg_coeff=0
# for p in dol_prob:
#     # print(p, optimal,(optimal -p)/Kull(p,optimal))
#     print((optimal -p))
#     if p!=optimal:
#         reg_coeff += (optimal -p)/Kull(p,optimal)
#     print(reg_coeff)
#     # print(p, reg_coeff)
# print(reg_coeff)
#
#
# # diff=0
# # for p in dol_prob:
# #     if p!=optimal:
# #         diff+=(optimal-p)
# #     # print(p, reg_coeff)
# # print(diff)
