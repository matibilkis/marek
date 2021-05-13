import numpy as np

class Helstrom():
    def __init__(self,alpha=0.4, ep=0.01):

        self.alpha = alpha
        self.beta = np.sqrt(ep)*alpha


    def overlap(self,a,b):
        return np.exp(-0.5*(a-b)**2)

    def __call__(self):
        alpha_plus = np.array([[1,0,0,0]])

        ##self.overlap <alpha_minus | v1>
        a = self.overlap(-self.alpha,self.alpha)
        #normalize v1
        a0=np.sqrt(1-a**2)
        alpha_minus = np.array([[a, a0,0,0]])

        ## self.overlap <beta | v1>
        b = self.overlap(self.beta, self.alpha)
        ## self.overlap <self.beta | v2>
        c = self.overlap(self.beta, -self.alpha ) - a*b
        ## normalize v12
        d = np.sqrt(1-b**2-c**2)
        beta_plus = np.array([[b,c,d,0]])

        ### <-beta | v1>
        nu = self.overlap(-self.beta,self.alpha)
        ## <-self.beta | v2>
        mu = self.overlap(-self.beta, -self.alpha) - a*self.overlap(-self.beta, self.alpha)
        ## <-self.beta| v3>
        xu = self.overlap(-self.beta, self.beta ) - c*mu - b*nu
        #normalize v4
        d4 = np.sqrt(1 - xu**2 - mu**2 - nu**2 )#+ (xu*d)**2 + (nu*a0)**2 + nu**2 )

        beta_minus = np.array([[nu,mu,xu,d4]])

        rho1 = (alpha_plus.T.dot(alpha_plus) + beta_plus.T.dot(beta_plus) )/2
        rho2 = (alpha_minus.T.dot(alpha_minus) + beta_minus.T.dot(beta_minus) )/2
        trace_norm = np.sum(np.abs(np.linalg.eigvals(rho1-rho2)))

        return (1 + trace_norm/2)/2

if __name__ == "__main__":
    hel = Helstrom(alpha=0.4, ep = 0.01)
    print(hel())
