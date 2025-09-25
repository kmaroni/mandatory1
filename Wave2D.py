import numpy as np
import sympy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
from matplotlib import cm

x, y, t = sp.symbols('x,y,t')

class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        x = np.linspace(0,1,N+1)
        self.xij, self.yij = np.meshgrid(x,x, indexing='ij', sparse=sparse)

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = 2, -5, 4, -1
        D[-1, -4:] = -1, 4, -5, 2
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        kx, ky = self.mx*np.pi, self.my*np.pi
        return self.c*np.sqrt(kx**2 + ky**2)

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx*sp.pi*x)*sp.sin(my*sp.pi*y)*sp.cos(self.w*t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        Un, Unm1 = np.zeros((2,N+1,N+1))
        Unm1[:] = sp.lambdify((x,y,t),self.ue(mx,my))(self.xij,self.yij,0)
        Un[:] = Unm1 + 0.5*(self.c*self.dt)**2*(self.D @ Unm1 + Unm1 @ self.D.T)
        return Un, Unm1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl*self.h/self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        ue = sp.lambdify((x,y,t),self.ue(self.mx,self.my))(self.xij,self.yij,t0)
        return np.sqrt(self.h**2*np.sum((u-ue)**2))

    def apply_bcs(self,U):
        """Return Unp1 with applied B.C.

        Parameters
        ----------
        U : array
            Solution of FDM without B.C.
        """
        U[0,:], U[-1,:], U[:,0], U[:,-1] = 0, 0, 0, 0
        return U

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (h, l2-error)
        """
        self.create_mesh(N)
        self.c = c
        self.cfl = cfl
        self.h = 1/N
        self.D = self.D2(N)/self.h**2
        self.mx, self.my = mx, my
        Un, Unm1 = self.initialize(N, mx, my)
        Unp1 = np.zeros((N+1,N+1))
        data = {0: Unm1.copy()}
        err = []
        for n in range(1,Nt):
            Unp1[:] = 2*Un - Unm1 + (self.c*self.dt)**2*(self.D @ Un + Un @ self.D.T)
            Unp1[:] = self.apply_bcs(Unp1)
            Unm1[:] = Un
            Un[:] = Unp1
            if store_data > 0 and n%store_data == 0:
                data[n] = Un.copy()

        if store_data > 0:
            return data
        elif store_data == -1:
            err.append(self.l2_error(Un,Nt*self.dt))
            return self.h, err

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for m in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i-1]/E[i])/np.log(h[i-1]/h[i]) for i in range(1, m+1, 1)]
        return r, np.array(E), np.array(h)

class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return modified second order differentiation matrix for Neumann B.C."""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N+1, N+1), 'lil')
        D[0, :4] = -2, 2, 0, 0
        D[-1, -4:] = 0, 0, 2, -2
        return D

    def ue(self, mx, my):
        """Return the exact standing wave for Neumann problem"""
        return sp.cos(mx*sp.pi*x)*sp.cos(my*sp.pi*y)*sp.cos(self.w*t)

    def apply_bcs(self,U):
        """Overide apply_bcs from Wave2D to do nothing. B.C. handeled by D2

        Parameters
        ----------
        U : array
            Solution of FDM with modified D2.
        """
        return U

def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 1e-2

def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1]-2) < 0.05

def test_exact_wave2d():
    mx = 2; my = mx
    cfl = 1/np.sqrt(2)
    sol = Wave2D()
    h, err = sol(750, 50, cfl=cfl, mx=mx, my=my, store_data=-1)
    try:
        assert err[-1] < 1e-12
    except AssertionError:
        print(f'Error of Dirichlet is {err[-1]}>1e-12.')

    solN = Wave2D_Neumann()
    h, err = solN(100, 50, cfl=cfl, mx=mx, my=my, store_data=-1)
    try:
        assert err[-1] < 1e-12
    except AssertionError:
        print(f'Error of Neumann is {err[-1]}>1e-12.')
