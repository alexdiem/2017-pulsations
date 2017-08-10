# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 09:03:24 2016

@author: alexandra
"""

from __future__ import division

import numpy as np
import matplotlib.pylab as plt
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate

plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = ['Arial']


# Units
s = 1
cm = 1e-2
um = 1e-4*cm
mM = 1e-3
uM = 1e-3 * mM
nM = 1e-6 * mM
C = 1
A = C/s
V = 1
mV = 1e-3 * V
F = C/V
pF = 1e-12 * F
S = A/V
pS = 1e-12 * S
dyn = 1
pa = 10 * dyn/cm**2
mmHg = 133.322 * pa


def bm_pressure(P, a, b, r):
    return -P * a**2/(b**2 - a**2) * (1 - b**2/r**2)
    
    
def bm_velocity(P, K0, K1, T, tc, L):
    nx = P.shape[1]
    nt = P.shape[0]
    U = np.zeros(nx)
    dP = np.zeros((nt, nx))
    dx = L/nx
    dt = T/nt
    dP = np.gradient(P, dx, axis=1)
    K = np.copy(dP)
    K[dP>=0] = K1
    K[dP<0] = K0
    t = np.linspace(0, T*tc, nt)
    U = -K*dP
    Uav = 1/T * np.trapz(U, x=t, dx=dt, axis=1)
    return U, Uav, dP
    
    
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
    ax.set_zlabel('pressure (mmHg)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fname = "plots/%s" % (fname)
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    
    
def contour_plot(fig_dims, Z, T, tc, L, N, fname):
    fig = plt.figure(figsize=fig_dims)
    x = np.linspace(T*(tc-1), T*tc, N)
    y = np.linspace(0, L, N)
    Y, X = np.meshgrid(x, y)
    plt.contourf(X, Y, Z, cmap=cm.viridis)
    plt.ylabel('t (s)')
    plt.xlabel('z (cm)')
    plt.colorbar()
    fname = "plots/%s" % (fname)
    plt.savefig(fname, dpi=600, bbox_inches='tight')
    
    
def x_plot(fig_dims, Z, T, tc, L, N, fname):
    colours = ['#377eb8', '#4daf4a', '#984ea3', '#d95f02']
    fig = plt.figure(figsize=fig_dims)
    x = np.linspace(0, L, Z.shape[0])
    pos = np.linspace(0, Z.shape[1]-1, 4)
    dt = T/Z.shape[1]
    for i in range(4):
        plt.plot(x, Z[int(pos[i]),:], label="t = %.2f" % (T*(tc-1)+pos[i]*dt),
                 color=colours[i], lw=2)
    plt.legend()
    fname = "plots/%s" % (fname)
    plt.savefig(fname, dpi=600, bbox_inches="tight")
    
    
def velocity_positions(fig_dims, fname):
    K01 = [-2.63920302915e-17,-2.28594860823e-17,-1.95762502338e-17,
           -1.65194031318e-17,-1.36685997102e-17,-1.1005730053e-17,
           -8.51463107071e-18,-6.18084065374e-18,-3.99138729716e-18,
           -1.93460946743e-18,0.0]
    K02 = [-1.33854374777e-17,-1.15938114024e-17,-9.92862885722e-18,
           -8.37826553499e-18,-6.93240288101e-18,-5.58185595778e-18,
           -4.31842721397e-18,-3.1347817965e-18,-2.02434085311e-18,
           -9.81189919231e-19,0.0]
    K03 = [-3.78844663939e-19,-3.28136722584e-19,-2.81007480623e-19,
           -2.37127938197e-19,-1.96206051847e-19,-1.57981862601e-19,
           -1.22223357232e-19,-8.87229392507e-20,-5.7294409051e-20,
           -2.77703710336e-20,0.0]
    K04 = [1.26277481498e-17,1.09375379573e-17,9.36661389598e-18,
           7.90400965859e-18,6.53999077732e-18,5.26589223258e-18,
           4.07398049951e-18,2.95733591799e-18,1.90975203501e-18,
           9.25649177164e-19,0.0]
    h = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    colours = ['#377eb8', '#4daf4a', '#984ea3', '#d95f02']
    fig = plt.figure(figsize=fig_dims)
    plt.plot(h, K01, label=r"K0/K1$ = 0.1", color=colours[0], lw=2)
    plt.plot(h, K02, label=r"K0/K1$ = 0.13", color=colours[1], lw=2)
    plt.plot(h, K03, label=r"K0/K1 = 0.16", color=colours[2], lw=2)
    plt.plot(h, K04, label=r"K0/K1 = 0.2", color=colours[3], lw=2)
    plt.legend(loc=4)
    fname = "plots/%s" % (fname)
    plt.savefig(fname, dpi=600, bbox_inches="tight")
    
    
def main():
    Ra = np.loadtxt("./data/Ra.csv", delimiter=',') * um
    Rb = np.loadtxt("./data/Rb.csv", delimiter=',') * um
    L = 1*cm
    R = (Rb-Ra)/2
    K0 = 1e-2 * 1e-10/1.5e-3 * um**2/(pa*s)
    K1 = 1.0 * 1e-10/1.5e-3 * um**2/(pa*s)
    T = 50 * s
    P = 60 * mmHg
    
    nt = P.shape[1]
    t = np.linspace(0, T, nt)
    
    Pbm = bm_pressure(P, Ra, Rb, R) 
    Ubm, Uav, dP = bm_velocity(Pbm, K0, K1, T, L) 
    
    print sum(Uav)/len(Uav)
    
    np.savetxt("./data/%s/u_bm.csv" % (sim), Ubm, delimiter=',')
    
    WIDTH = 510  # the number latex spits out
    FACTOR = 1.0  # the fraction of the width you'd like the figure to occupy
    fig_width_pt  = WIDTH * FACTOR
    inches_per_pt = 1.0 / 72.27
    golden_ratio  = (np.sqrt(5) - 1.0) / 2.0  # because it looks good
    fig_width_in  = fig_width_pt * inches_per_pt  # figure width in inches
    fig_height_in = fig_width_in * golden_ratio   # figure height in inches
    fig_dims    = [fig_width_in, fig_height_in] # fig dims as a list
    
    #p3d_plot(fig_dims, Pbm, T, tc, L, N, "bm_pressure0.png")
    #contour_plot(fig_dims, Ubm, T, tc, L, N, "bm_vel_no_valve.png")

    #x_plot(fig_dims, dP, T, tc, L, N, "bm_pgrad_x.png")    
    #x_plot(fig_dims, Ubm, T, tc, L, N, "bm_vel_x.png")  

    #velocity_positions(fig_dims, "K0K1_vs_h.png")      
        
    
if __name__ == "__main__":
    main()
