import numpy as np
from misc import Kull
import basics
b = basics.Basics(amplitude=0.3968)
print((b.P(-.3968, -.7,1,1) + b.P(.3968, -.7,1,0))/2)
print(b.P(-.3968, -.7,1,0),b.err_dolinar(-.7))
# probs = np.arange(1,10)/10
# print(probs, len(probs))
# r=0
# for i in probs:
#     if i != 0.9:
#         r+=(0.9-i)/Kull(i,0.9)
# print(r)
# print(np.log(10**4)*7.5)
