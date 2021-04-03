# modified from matplotlib documentation example
# https://matplotlib.org/2.0.2/examples/axes_grid/demo_parasite_axes2.html

"""
Parasite axis demo

The following code is an example of a parasite axis. It aims to show a user how
to plot multiple different values onto one single plot. Notice how in this
example, par1 and par2 are both calling twinx meaning both are tied directly to
the x-axis. From there, each of those two axis can behave separately from the
each other, meaning they can take on separate values from themselves as well as
the x-axis.
"""
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path

input_data = pd.read_csv('random_forest_regression_565_rpm.csv')

fig = plt.figure(figsize=(10, 8))
host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.75)

par1 = host.twinx()
par2 = host.twinx()

offset = 60
new_fixed_axis = par2.get_grid_helper().new_fixed_axis
par2.axis["right"] = new_fixed_axis(loc="right",
                                    axes=par2,
                                    offset=(offset, 0))

par2.axis["right"].toggle(all=True)

SEED_CRYSTAL_MASS = 'SEED_CRYSTAL_MASS'
YIELD = 'YIELD'
GROWTH_RATE = 'GROWTH_RATE'
MEAN_DIAMETER = 'MEAN_DIAMETER'

host.set_xlabel(SEED_CRYSTAL_MASS)
host.set_ylabel(YIELD)
par1.set_ylabel(GROWTH_RATE)
par2.set_ylabel(MEAN_DIAMETER)

x_ax = input_data['SEED_CRYSTAL_MASS']
p1, = host.plot(x_ax, input_data[YIELD], label=YIELD)
p2, = par1.plot(x_ax, input_data[GROWTH_RATE], label=GROWTH_RATE)
p3, = par2.plot(x_ax, input_data[MEAN_DIAMETER], label=MEAN_DIAMETER)

# host.legend()

host.axis["left"].label.set_color(p1.get_color())
par1.axis["right"].label.set_color(p2.get_color())
par1.axis['right'].toggle(ticklabels=True, label=True)
par2.axis["right"].label.set_color(p3.get_color())
plt.title('Random Forest Optimal Conditions for 565 rpm Agitation Rate')
plt.draw()
# approximated intersection point
plt.axvline(x=0.5643, color='k')


cur_path = str(Path(__file__).parents[0])
plt.savefig(os.path.join(cur_path, 'results/565rpm_optimization_seed_crystal_mass.png'))