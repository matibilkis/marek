# Marek: the RL agent that achieves optimal discrimination rates from scratch

This is the code that led to the results presented in "Real-time calibration of coherent-state receivers: learning by trial and error".
Marek's objective is to learn the optimal discrimination strategy over an unknown quantum-classical channel; we frame this as a reinforcement learning problem.

## The setup:
The goal is to calibrate the following receiver:
![alt text](https://github.com/matibilkis/marek/blob/master/ploting_programs/receiver.png),
departing from complete ignorance of any experimental details. As explained in the paper, the model-free learning of such a receiver permits optimal success rate over noisy channels, in which dark counts, phase flips or energy shifts may occur.

For instance, this kind of learning curves are obtained:

![alt text](https://github.com/matibilkis/marek/blob/master/ploting_programs/17jan_enh-QLexp.png).

## Find the paper!: ![here](https://github.com/matibilkis/marek/blob/master/paper_preprint.pdf),

