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
