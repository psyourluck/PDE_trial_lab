## plot the figure

from math import log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

node_spacing_box = []
elltwo_error_box = np.load('tri1_ell2.npy')
sup_error_box = np.load('tri1_sup.npy')
H1_error_box = np.load('tri1_h1.npy')


for n in range(2,21):
    node_spacing_box.append(1 / float(n))

log_node_spacing_box = []
for i in node_spacing_box:
    log_node_spacing_box.append(-log(i))



log_elltwo_error_box = []
for i in elltwo_error_box:
    log_elltwo_error_box.append(log(i))

log_sup_error_box = []
for i in sup_error_box:
    log_sup_error_box.append(log(i))

# log_energy_error_box = []
# for i in energy_error_box:
#     log_energy_error_box.append(log(i))

log_H1_error_box = []
for i in H1_error_box:
    log_H1_error_box.append(log(i))

sup_rate = np.polyfit(np.array(log_node_spacing_box), np.array(log_sup_error_box), 1)[0]
elltwo_rate = np.polyfit(np.array(log_node_spacing_box), np.array(log_elltwo_error_box), 1)[0]
h1_rate = np.polyfit(np.array(log_node_spacing_box), np.array(log_H1_error_box), 1)[0]
print('sup rate: {:.4f}, elltwo rate: {:.4f}, h1 rate: {:.4f}'.format(sup_rate, elltwo_rate, h1_rate))

# print "The Sup Convergence Exponent is ", sup_rate
# print "The L2 Convergence Exponent is ", elltwo_rate

plt.plot(log_node_spacing_box, log_sup_error_box, color = 'blue')
plt.plot(log_node_spacing_box, log_elltwo_error_box, color = 'green')
plt.plot(log_node_spacing_box, log_H1_error_box, color='red')

blue_patch = mpatches.Patch(color='blue', label='Sup Error')
green_patch = mpatches.Patch(color='green', label = 'L2 Error')
red_patch = mpatches.Patch(color='red', label = 'H1 Error')
plt.legend(handles=[blue_patch, green_patch, red_patch])


plt.xlabel('log(1/n)')
plt.ylabel('log(error)')
plt.title('log-log Error Plot')


plt.show()