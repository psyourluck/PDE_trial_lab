## plot the figure

from math import log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

node_spacing_box = []
elltwo_error_box = np.load('LW_2.npy')
sup_error_box = np.load('LW_sup.npy')

elltwo_error_box = elltwo_error_box[1:]
sup_error_box = sup_error_box[1:]

# for n in range(6):
#     node_spacing_box.append(1 / float(n))

log_node_spacing_box = []
for i in range(5):
    log_node_spacing_box.append(log(2)*(i+3))



log_elltwo_error_box = []
for i in elltwo_error_box:
    log_elltwo_error_box.append(log(i))

log_sup_error_box = []
for i in sup_error_box:
    log_sup_error_box.append(log(i))

# log_energy_error_box = []
# for i in energy_error_box:
#     log_energy_error_box.append(log(i))

# log_H1_error_box = []
# for i in H1_error_box:
#     log_H1_error_box.append(log(i))

sup_rate, sup_co = np.polyfit(np.array(log_node_spacing_box), np.array(log_sup_error_box), 1)
elltwo_rate, elltwo_co = np.polyfit(np.array(log_node_spacing_box), np.array(log_elltwo_error_box), 1)
#h1_rate = np.polyfit(np.array(log_node_spacing_box), np.array(log_H1_error_box), 1)[0]
print('sup rate: {:.4f}, elltwo rate: {:.4f}'.format(sup_rate, elltwo_rate))
sup_C = np.exp(sup_co)/(1-2**sup_rate)
elltwo_C = np.exp(elltwo_co)/(1-2**elltwo_rate)
sup_err = sup_C * 100**sup_rate
elltwo_err = elltwo_C * 100**elltwo_rate
print('sup error: {:.6f}, elltwo rate: {:.6f}'.format(sup_err, elltwo_err))


# print "The Sup Convergence Exponent is ", sup_rate
# print "The L2 Convergence Exponent is ", elltwo_rate

plt.plot(log_node_spacing_box, log_sup_error_box, color = 'blue')
plt.plot(log_node_spacing_box, log_elltwo_error_box, color = 'green')
#plt.plot(log_node_spacing_box, log_H1_error_box, color='red')

blue_patch = mpatches.Patch(color='blue', label='Sup Error')
green_patch = mpatches.Patch(color='green', label = 'L2 Error')
#red_patch = mpatches.Patch(color='red', label = 'H1 Error')
plt.legend(handles=[blue_patch, green_patch])


plt.xlabel('log(1/n)')
plt.ylabel('log(error)')
plt.title('log-log Error Plot')


plt.show()