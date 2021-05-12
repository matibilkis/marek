import numpy as np
import cmath
import os
import pickle


class Record():
    def __init__(self, FolderName):

        if not os.path.exists("bounds_optimals_and_limits"):
            os.makedirs("bounds_optimals_and_limits")
        if not os.path.exists(FolderName):
            os.makedirs(FolderName)
        os.chdir(FolderName)
        if not os.path.exists("figures"):
            os.makedirs("figures")
        if not os.path.exists("number_rune.txt"):
            with open("number_rune.txt", "w+") as f:
                f.write("0")
                f.close()

        self.number_run = self.record()
    #

    def record(self):
        with open("number_rune.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        with open("number_rune.txt", "w") as f:
            f.truncate(0)
            f.write(str(int(a)+1))
            f.close()
        if not os.path.exists("run_"+str(int(a)+1)):
            os.makedirs("run_"+str(int(a)+1))
        os.chdir("run_"+str(int(a)+1)) #I leave you in this directory
        os.makedirs("tables")
        return int(a)+1


class Complex(complex):
    def __repr__(self):
        rp = '%7.5f' % self.real if not self.pureImag() else ''
        ip = '%7.5fj' % self.imag if not self.pureReal() else ''
        conj = '' if (
            self.pureImag() or self.pureReal() or self.imag < 0.0
        ) else '+'
        return '0.0' if (
            self.pureImag() and self.pureReal()
        ) else rp + conj + ip

    def pureImag(self):
        return abs(self.real) < 1e-14

    def pureReal(self):
        return abs(self.imag) < 1e-14
