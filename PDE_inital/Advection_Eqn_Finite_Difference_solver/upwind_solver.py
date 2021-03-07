import numpy as np
import matplotlib.pyplot as plt


class UpwindMethod:
    def __init__(self, N, tmax, dt, prb, yinf=0, ysup=1, dynamic=True, xmin=0, exact=True):
        self.N = N  # number of nodes
        self.tmax = tmax
        self.xmin = xmin
        self.xmax = 1
        self.dt = dt  # timestep
        self.v = 1  # velocity
        self.prb = prb
        self.k = 80
        self.yinf = yinf
        self.ysup = ysup
        self.dynmaic = dynamic
        self.exact = exact
        self.initializeDomain()
        self.initializeU()
        self.initializeParams()

    def initializeDomain(self):
        self.dx = (self.xmax - self.xmin) / (self.N - 1)
        self.x = np.arange(self.xmin, self.xmax+self.dx, self.dx)

    def initializeU(self):
        if self.prb == 0:
            u0 = np.exp(-200 * (self.x - self.xc) ** 2)
            self.u = u0.copy()
            self.unp1 = u0.copy()
        elif self.prb == 1:
            u0 = (self.x <= 0) + 0.
            self.u = u0.copy()
            self.unp1 = u0.copy()
        elif self.prb == 2:
            u0 = np.exp(-100 * (self.x - 0.5) ** 2) * np.sin(self.k * self.x)
            self.u = u0.copy()
            self.unp1 = u0.copy()
        elif self.prb == 3:
            xnp = (self.x - 0.3) * (self.x >= -0.7) + (self.x + 1.7) * (self.x < -0.7)
            u0 = -xnp * np.sin(1.5 * np.pi * xnp ** 2) * (xnp <= -1 / 3) + \
                 abs(np.sin(2 * np.pi * xnp)) * (abs(xnp) < 1 / 3) + \
                 (2 * xnp - 1 - np.sin(3 * np.pi * xnp) / 6) * (xnp >= 1 / 3)
            self.u = u0.copy()
            self.unp1 = u0.copy()
        elif self.prb == 4:
            u0 = np.sin(2*np.pi*self.x)
            self.u = u0.copy()
            self.unp1 = u0.copy()


    def initializeParams(self):
        self.nsteps = round(self.tmax / self.dt)
        self.alpha = self.v * self.dt / (self.dx)
        self.alpha1 = np.pi * self.dt / (self.dx)

    def upwind_step(self, a, b):
        if a+b > 0:
            return a**2
        else:
            return b**2


    def solve_and_plot(self, plot=True):
        tc = 0

        for i in range(self.nsteps-1):
            plt.clf()
            tc += self.dt

            # The FOU scheme, Eq. (18.21)
            if self.prb != 4:
                for j in range(1, self.N):
                    self.unp1[j] = self.u[j] - self.alpha * (self.u[j] - self.u[j - 1])
            else:
                for j in range(1, self.N-1):
                    self.unp1[j] = self.u[j] - self.alpha1 * (self.upwind_step(self.u[j], self.u[j+1]) - \
                                                              self.upwind_step(self.u[j-1], self.u[j]))


            if self.prb == 0:
                # Periodic BCs
                self.unp1[self.N + 1] = self.unp1[0]
                self.unp1[self.N + 2] = self.unp1[1]
                uexact = np.exp(-200 * (self.x - self.xc - self.v * tc) ** 2)
            elif self.prb == 1:
                ## inflow boundary condition
                # self.u[0] = 1
                self.unp1[0] = 1
                ## we use upwind scheme for the rightmost grid point
                # self.unp1[self.N-1] = self.u[self.N - 1] - 2*self.alpha * (self.u[self.N-1] - self.u[self.N-2])
                uexact = (self.x - self.v * tc <= 0) + 0.
            elif self.prb == 2:
                ## periodic BC
                self.unp1[0] = self.unp1[self.N-1]
                num = (self.x - self.v * tc) - np.floor(self.x - self.v * tc)
                uexact = np.exp(-100 * (num - 0.5) ** 2) * np.sin(self.k * num)
            elif self.prb == 3:
                ## periodic BC
                self.unp1[0] = self.unp1[self.N-1]
                num = (self.x - self.v * tc) - np.floor((self.x - self.v * tc + 1) / 2) * 2
                xnp = (num - 0.3) * (num >= -0.7) + (num + 1.7) * (num < -0.7)
                uexact = -xnp * np.sin(1.5 * np.pi * xnp ** 2) * (xnp <= -1 / 3) + \
                         abs(np.sin(2 * np.pi * xnp)) * (abs(xnp) < 1 / 3) + \
                         (2 * xnp - 1 - np.sin(3 * np.pi * xnp) / 6) * (xnp >= 1 / 3)
            elif self.prb == 4:
                self.unp1[0] = self.u[0] - self.alpha1 * (self.upwind_step(self.u[0], self.u[1]) - \
                                                          self.upwind_step(self.u[self.N-2], self.u[0]))
                self.unp1[self.N-1] = self.unp1[0]
            self.u = self.unp1.copy()


            if self.dynmaic == True:
                plt.plot(self.x, self.u, 'bo-', label="First-order Upwind")
                if self.exact == True:
                    plt.plot(self.x, uexact, 'r', label="Exact solution")
                plt.axis((self.xmin, self.xmax, self.yinf, self.ysup))
                plt.grid(True)
                plt.xlabel("Distance (x)")
                plt.ylabel("u")
                plt.legend(loc=1, fontsize=12)
                plt.suptitle("Time = %1.3f" % (tc + self.dt))
                plt.pause(0.01)

        if plot == True:
            plt.plot(self.x, self.u, 'bo-', label="First-order Upwind")
            if self.exact == True:
                plt.plot(self.x, uexact, 'r', label="Exact solution")
            plt.axis((self.xmin, self.xmax, self.yinf, self.ysup))
            plt.grid(True)
            plt.xlabel("Distance (x)")
            plt.ylabel("u")
            plt.legend(loc=1, fontsize=12)
            plt.suptitle("Time = %1.3f" % (tc + self.dt))
            plt.pause(0.01)



def main():
    #sim = UpwindMethod(101, 0.5, 0.005, 1, yinf=-0.3, ysup=1.4, dynamic=False)
    # sim = UpwindMethod(401, 0.5, 0.00125, 1, yinf=-0.3, ysup=1.4, dynamic=False)
    # sim = UpwindMethod(201, 8, 0.004, 2, yinf=-1, ysup=1, dynamic=False)
    #sim = UpwindMethod(501, 8, 0.0032, 3, yinf=-1, ysup=1, dynamic=False, xmin=-1)
    # sim = UpwindMethod(1025, 0.015, 0.001, 4, yinf=-1, ysup=1, dynamic=False, exact=False)
    #
    # sim.solve_and_plot()
    # plt.show()

    # ## error estimate
    # sim = UpwindMethod(5, 0.015, 0.001, 4, yinf=-1, ysup=1, dynamic=False, exact=False)
    # sim.solve_and_plot(plot=False)
    elltwo_error_box = []
    sup_error_box = []
    node_spacing_box = []
    for i in range(2,11):
        n = int(np.exp2(i))
        dt = 0.001
        sim = UpwindMethod(n+1, 0.015, dt, 4, yinf=-1, ysup=1, dynamic=False, exact=False)
        sim.solve_and_plot(plot=False)
        if i == 2:
            temp = sim.u
            continue
        temp_re = sim.u[::2]
        err = temp_re - temp
        ## L2 error
        elltwo_err = ((err**2).sum()*2/n)**0.5
        elltwo_error_box.append(elltwo_err)
        ## sup error
        sup_err = max(abs(err))
        sup_error_box.append(sup_err)
        print('grid number: {:d}, elltwo_error: {:.6f}, sup_error: {:.6f}'.format(n, elltwo_err, sup_err))
        temp = sim.u
        if n == 256:
            dt /= 2
    np.save('upwind_2', np.array(elltwo_error_box))
    np.save('upwind_sup', np.array(sup_error_box))


if __name__ == "__main__":
    main()

# N = 100
# tmax = 2.5 # maximum value of t