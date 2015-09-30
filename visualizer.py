import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

###ROUGH PROTOTYPE OF PLOTTING MECHANISM###
filename = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
line_num = 1036;





skiphead = 0;
skipend = 0
u = np.genfromtxt(filename, delimiter = ",", skip_header = skiphead, skip_footer = skipend, deletechars = "\n");
x = y = range(0,200)
X, Y = np.meshgrid(x,y);
fig = plt.figure()
#ax = Axes3D(fig)
#ax.autoscale(False)
for i in range(start, end):
	ax = fig.add_subplot(111, projection='3d')
	ax.set_zlim(0, 2)
	z = u[i,0:40000];
	Z = z.reshape(X.shape)
	ax.plot_surface(X, Y, Z)
#	cset = ax.plot_surface(X,Y,Z,16)
#	ax.clabel(cset, fontsize=9, inline = 1)
	name = "images/img" + str(i)
	plt.savefig(name)
	plt.delaxes(ax)











