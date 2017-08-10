from __future__ import division

import sys
import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp2d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

colours = ['#377eb8', '#4daf4a', '#984ea3', '#d95f02']

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'stixsans'


def grid(x0, L, nx):
    x = np.linspace(x0, L, nx)
    dx = x[1] - x[0]
    return x, dx


def cfl_condition(dt, dx, v):
    d = dt/dx
    return True if (d <= np.power(np.absolute(v),2)).all() == True else False
    
    
def initial_conditions(Ri):
    U = Ri * 2e-5
    return U
    
    
def F(U, ubm):
    return U*ubm
    

def S(U):
    return 0
    
    
def lax_wendroff(U_prev, F_prev, S_prev, u_bm, dt, dx):
    # u_prev = [U[m-1], U[m], U[m+1]], a_prev, p_prev analogously
    U_np_mp = (U_prev[2]+U_prev[1])/2 + dt/2 * (-(F_prev[2]-F_prev[1])/dx +\
                (S_prev[2]+S_prev[1])/2)
    U_np_mm = (U_prev[1]+U_prev[0])/2 + dt/2 * (-(F_prev[1]-F_prev[0])/dx +\
                (S_prev[1]+S_prev[0])/2)
                
    F_np_mp = F(U_np_mp, u_bm)
    F_np_mm = F(U_np_mm, u_bm)
    S_np_mp = S(U_np_mp)
    S_np_mm = S(U_np_mm)
    
    U_np = U_prev[1] - dt/dx * (F_np_mp-F_np_mm) + dt/2 * (S_np_mp+S_np_mm)
    return U_np
    
    
def numerical(U, ubm, time, dt, dx, x, L):
    for i in range(1,len(time)):
        # test cfl condition
        v = (max(U[i-1,:])) 
#        if cfl_condition(dt, dx, v) == False:
#            raise ValueError(
#                'Time step dt = %e is too large. CFL condition not fulfilled.'\
#                % (dt))
        
        # inlet boundary condition
        U[i,0] = U[0,0]
                
        for j in range(1,len(x)-1):
            u_prev = U[i-1,j-1:j+2]
            f_prev = u_prev * ubm[i-1,j-1:j+2]
            s_prev = np.array([0,0,0])
            if len(u_prev) == 2: # at the end of the array
                u_prev = U[i-1,j-1:]
                f_prev = u_prev * ubm[i-1,j-1:]
                s_prev = np.array([0,0,0])
            U[i,j] = lax_wendroff(u_prev, f_prev, s_prev, ubm[i,j], dt, dx)                                                                    
        # outlet boundary condition
        U[i,-1] = U[0,-1]
    
    return U
    
    
def contour_plot(fig_dims, Z, T, tc, L, N, fname):
    fig = plt.figure(figsize=fig_dims)
    x = np.linspace(T*(tc-1), T*tc, N)
    y = np.linspace(0, L, N)
    Y, X = np.meshgrid(x, y)
    plt.contourf(X, Y, Z-1e-6, cmap=cm.viridis)
    plt.ylabel('t (s)')
    plt.xlabel('z (cm)')
    plt.colorbar()
    fname = "plots/%s" % (fname)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    
    
def p3d_plot(fig_dims, Z, T, tc, L, N, fname):
    fig = plt.figure(figsize=fig_dims)
    ax = fig.gca(projection='3d')
    x = np.linspace(T*(tc-1), T*tc, N)
    y = np.linspace(0, L, N)
    Y, X = np.meshgrid(y, x)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.viridis,
                  linewidth=0, antialiased=False)
    ax.set_xlabel('t (s)')
    ax.set_ylabel('z (cm)')
    ax.set_zlabel('displacement (um)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fname = "plots/%s" % (fname)
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    
    
def h_plot(fig_dims, Z, L, N, dt, T, fname):
    fig = plt.figure(figsize=fig_dims)
    x = np.linspace(0, L, N)
    c = 10
    times = np.linspace(0, N-1, c, dtype=int)
    i = 0
    colors = cm.viridis(np.linspace(0,1,c))
    for t in times:
        plt.plot(x, Z[t,:], label="t=%.2fs" % (T+t*dt), color=colors[i])
        i += 1
    plt.xlabel('z (cm)')
    plt.ylabel('h (nm) + 200 nm')
    plt.legend()
    fname = "plots/%s" % (fname)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    
    
def flux_positions(fig_dims, fname):
    K01 = np.array([-2.63920302915e-17,-2.28594860823e-17,-1.95762502338e-17,
           -1.65194031318e-17,-1.36685997102e-17,-1.1005730053e-17,
           -8.51463107071e-18,-6.18084065374e-18,-3.99138729716e-18,
           -1.93460946743e-18,0.0])
    K02 = np.array([-1.33854374777e-17,-1.15938114024e-17,-9.92862885722e-18,
           -8.37826553499e-18,-6.93240288101e-18,-5.58185595778e-18,
           -4.31842721397e-18,-3.1347817965e-18,-2.02434085311e-18,
           -9.81189919231e-19,0.0])
    K03 = np.array([-3.78844663939e-19,-3.28136722584e-19,-2.81007480623e-19,
           -2.37127938197e-19,-1.96206051847e-19,-1.57981862601e-19,
           -1.22223357232e-19,-8.87229392507e-20,-5.7294409051e-20,
           -2.77703710336e-20,0.0])
    K04 = np.array([1.26277481498e-17,1.09375379573e-17,9.36661389598e-18,
           7.90400965859e-18,6.53999077732e-18,5.26589223258e-18,
           4.07398049951e-18,2.95733591799e-18,1.90975203501e-18,
           9.25649177164e-19,0.0])
    h = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    colours = cm.viridis(np.linspace(0,1,4))
    fig = plt.figure(figsize=fig_dims)
    plt.plot(h, K01*1e12, label=r"K0/K1 = 2.68e-2", color=colours[0], lw=2)
    plt.plot(h, K02*1e12, label=r"K0/K1 = 2.70e-2", color=colours[1], lw=2)
    plt.plot(h, K03*1e12, label=r"K0/K1 = 2.72e-2", color=colours[2], lw=2)
    plt.plot(h, K04*1e12, label=r"K0/K1 = 2.74e-2", color=colours[3], lw=2)
    plt.legend(loc=4)
    plt.xlabel('e')
    plt.ylabel('q (um^3/s)')
    fname = "plots/%s" % (fname)
    plt.savefig(fname, dpi=600, bbox_inches="tight")


def main():
    WIDTH = 510  # the number latex spits out
    FACTOR = 0.6  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list    
    
    # set up all variables
    sim = "mca_paper8"
    P = np.loadtxt("./data/%s/p0_mca_paper8.csv" % (sim), delimiter=',')
    ubm = np.loadtxt("./data/%s/u_bm.csv" % (sim), delimiter=',')
    
    R = 0.14
    h = 0.5*R
    a = R
    b = R+h
    r = R + 0.5*h
    k1 = 2.0e7
    k2 = -22.53
    k3 = 8.65e5
    Ehr = k1 * np.exp(k2*R) + k3
    E = Ehr * R/h
    nu = 0.49
    lam = E*nu/((1+nu)*(1-2*nu))
    mu = E/(2*(1+nu))
    T = 0.85
    tc = 8
    L = a*50
    
    Ri = a + P*a**2/(2*(b**2-a**2)) * (r/(lam*mu) + b**2/(mu*r))
    nt = P.shape[1]
    nx = P.shape[0]
    x = np.linspace(T*(tc-1), T*tc, nt)
    y = np.linspace(0, L, nx)
    dt = x[1] - x[0]
    dx = y[1] - y[0]
#    f = interp2d(x, y, Ri, kind='cubic')
#    N = min(P.shape)
#    x = np.linspace(T*(tc-1), T*tc, N)
#    y = np.linspace(0, L, N)
#    Ri = f(x, y)    
#    p3d_plot(fig_dims, Ri, T, tc, L, N, "ri.png")
#    sys.exit()
    
    # initial condition
    U = initial_conditions(Ri)
    u0 = U[0,:]
    
    f = interp2d(x, y, U, kind='linear')
    g = interp2d(x, y, Ri, kind='linear')
    N = min(P.shape)
    x = np.linspace(T*(tc-1), T*tc, N)
    y = np.linspace(0, L, N)
    U = f(x, y)
    Ri = g(x, y)
    U = numerical(U, ubm, y, dt, dx, x, L)
    
    q = np.zeros((N, N))
    qbar = 0.0
    for i in range(U.shape[0]):
        q[i,:] = U[i,:] * ubm[i,:]
        qbar += sum(q[i,:])
    
    print (qbar/N**2)
    
    #H = (U/Ri)*1e7
    #dH = np.gradient(H, 0.1, 1e-5)[0]
    #h_plot(fig_dims, H, L, N, dt, T*(tc-1), "h.png")
    #h_plot(fig_dims, H, L, N, dt, T*(tc-1), "h.png")
    #p3d_plot(fig_dims, H, T, tc, L, N, "h3d.png")
    #p3d_plot(fig_dims, (Ri-a)*1e5, T, tc, L, N, "Ri.png")
    #flux_positions(fig_dims, "K0K1.png")
    
    
        
    
if __name__ == "__main__":
    main()
