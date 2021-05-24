import numpy as np

class Basics():
    def __init__(self, amplitude = 0.4, number_phases=2, epsilon=.01):
        self.number_phases=number_phases
        self.possible_phases = self.roots()
        self.epsilon = epsilon
        self.amplitude = amplitude
    def roots(self):
        return np.array([np.exp(2*np.pi*k*1j/self.number_phases) for k in range(self.number_phases)])

    def insert(self,v,M):
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

    def outcomes_universe(self,L):
        a = np.array([0,1])
        two_outcomes = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(int)
        if L<2:
            return np.array([0,1]).astype(int)
        elif L==2:
            return two_outcomes
        else:
            x = self.insert(a,two_outcomes)
            for i in range(L-3):
                x = self.insert(a,x)
            return x.astype(int)

    def P(self,a,b,et,n):
        p0=np.exp(-abs((et*a)+b)**2)
        if n ==0:
            return p0
        else:
            return 1-p0


    def success_probability_1L(self,betas):
        p=0
        b0=betas[0]
        for n1 in [0,1]:
            ## best guess ?
            pguess = [0,0]
            for arg,ph in enumerate(self.possible_phases):
                for att in [self.epsilon, 1]:
                    pguess[arg]+=self.P(np.sqrt(att)*ph*self.amplitude, b0 ,1, n1)/4
            ph = self.possible_phases[np.argmax(pguess)]
            for att in [self.epsilon, 1]:
                p+=self.P(np.sqrt(att)*ph*self.amplitude, b0 ,1, n1)/4
        return -p



    def success_probability_2L(self,betas):
        b0, b1 ,b2 = betas
        p=0
        for n1,n2 in zip(*self.outcomes_universe(2).T):
            ## best guess ?
            pguess = [0,0]
            for arg,ph in enumerate(self.possible_phases):
                for att in [self.epsilon, 1]:
                    pguess[arg]+=self.P(np.sqrt(att)*ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(np.sqrt(att)*ph*self.amplitude, [b1,b2][n1] ,1/np.sqrt(2), n2)/4
            ph = self.possible_phases[np.argmax(pguess)]
            for att in [self.epsilon, 1]:
                p+=self.P(np.sqrt(att)*ph*self.amplitude, b0 ,1/np.sqrt(2), n1)*self.P(np.sqrt(att)*ph*self.amplitude, [b1,b2][n1] ,1/np.sqrt(2), n2)/4
        return -p

    def success_probability_3L(self,betas):
        b0, b1 ,b2,b3, b4, b5, b6 = betas
        p=0
        for n1,n2,n3 in zip(*self.outcomes_universe(3).T):
            ## best guess ?
            pguess = [0,0]
            for arg,ph in enumerate(self.possible_phases):
                for att in [self.epsilon, 1]:
                    pguess[arg]+=self.P(np.sqrt(att)*ph*self.amplitude, b0 ,1/np.sqrt(3), n1)*self.P(np.sqrt(att)*ph*self.amplitude, [b1,b2][n1] ,1/np.sqrt(3), n2)*self.P(ph*np.sqrt(att)*self.amplitude, np.array([[b3,b4],[b5,b6]])[n1,n2] ,1/np.sqrt(3), n3) /4
            ph = self.possible_phases[np.argmax(pguess)]
            for att in [self.epsilon, 1]:
                p+=self.P(np.sqrt(att)*ph*self.amplitude, b0 ,1/np.sqrt(3), n1)*self.P(np.sqrt(att)*ph*self.amplitude, [b1,b2][n1] ,1/np.sqrt(3), n2)*self.P(ph*np.sqrt(att)*self.amplitude, np.array([[b3,b4],[b5,b6]])[n1,n2] ,1/np.sqrt(3), n3) /4
        return -p
