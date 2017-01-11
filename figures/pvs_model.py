from __future__ import division
from matplotlib.pyplot import plot, annotate, arrow, axhline, xlim, ylim, xlabel, ylabel, xticks, yticks, savefig, show, figure, text
from matplotlib import rc
import numpy as np
import os

rc('font', family='sans-serif', size=9)
rc('text', usetex=True)

pi = np.pi
end = 80*pi
mid = end/2
eta = np.linspace(0, end, 40*256, endpoint=True)
Hi = 0.05*np.sin(0.05*eta)+0.99
Ho = 0.05*np.sin(0.05*(eta+20))+1.3
Hb = 0.05*np.sin(0.05*(eta+5))+0.5

arrow_start = mid-20

def cm2inch(value):
    return value/2.54

fig = figure(figsize=(cm2inch(17.2), cm2inch(8.7)))

plot(eta, Hi, linewidth=2, color='k')
plot(eta, Ho, linewidth=2, color='k')
plot(eta, Hb, linewidth=2, color='k')

axhline(y=0.99, linewidth=1, linestyle='dashed', color='k')
axhline(y=1.3, linewidth=1, linestyle='dashed', color='k')
axhline(y=0.5, linewidth=1, linestyle='dashed', color='k')

annotate('artery wall', (mid,0.7), xycoords='data', xytext=(-22, 0), textcoords='offset points')
annotate('ISF flow', (mid,1.15), xycoords='data', xytext=(-17,0.05), textcoords='offset points')
annotate('artery wall', (mid,1.4), xycoords='data', xytext=(-22,0), textcoords='offset points')
annotate('blood flow', (mid,0.2), xycoords='data', xytext=(-20,0), textcoords='offset points')
annotate(r'$h(z,t)$', (50,1.11), xycoords='data', xytext=(5,0), textcoords='offset points')

arrow(arrow_start, 0.15, 40, 0, head_width=0.05, head_length=3, color='k')
arrow(arrow_start+40, 1.1, -40, 0, head_width=0.05, head_length=3, color='k')
arrow(60, 1.09, 0, -0.06, head_width=2, head_length=0.02, color='k')
arrow(60, 1.17, 0, 0.06, head_width=2, head_length=0.02, color='k')

text(20, 1.06, r'$R_i(z,t)$')
text(20, 1.36, r'$R_o(z,t)$')
text(20, 0.57, r'$R(z,t)$')

xlim(0,end)
ylim(0,1.5)
xlabel(r'$z$')
ylabel(r'$r$', rotation='horizontal')
xticks([0, end], [r'0', r'$L$'])
yticks([0.5, 0.99, 1.3], [r'$R$', r'$a$', r'$b$'])
savefig('./pvs_model.pdf')
show()
