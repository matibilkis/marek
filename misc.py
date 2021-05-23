import numpy as np
import cmath
import os
import pickle

def insert(v,M):
    """
    Takes v, M and returns an array that has, for each element of v, a matrix M

    Example:
    x = [x0,x1]
    y = [[0,0],[0,1],[1,0],[1,1]]
    insert(x,y) returns

    [x0 0 0]
    [x0 0 1]
    [x0 1 0]
    [x0 1 1]
    [x1 0 0]
    [x1 0 1]
    [x1 1 0]
    [x1 1 1]
    """
    try:
        a=M.shape
        if len(a)<2:
            a.append(1)
    except Exception:
         a = [1,len(M)]
    result=np.zeros((a[0]*len(v),a[1] +1 )).astype(int)

    f = len(v)+1
    cucu=0
    for k in v:
        result[cucu:(cucu+a[0]),0] = k
        result[cucu:(cucu+a[0]),1:] = M
        cucu+=a[0]
    return result



def outcomes_universe(L):
    """
    Takes L (# of photodetections in the experiment) and returns
    all possible outcomes in a matrix of 2**L rows by L columns,
    which are all possible sequence of outcomes you can ever get.
    """
    a = np.array([0,1])
    two_outcomes = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(int)
    if L<2:
        return np.array([0,1]).astype(int)
    elif L==2:
        return two_outcomes
    else:
        x = insert(a,two_outcomes)
        for i in range(L-3):
            x = insert(a,x)
        return x.astype(int)

class Record():
    def __init__(self, FolderName):

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
