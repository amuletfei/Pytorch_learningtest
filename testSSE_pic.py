import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D

x = np.arange(-1, 3, 0.05)
y = np.arange(-1, 3, 0.05)
a, b = np.meshgrid(x, y)
SSE =(2-a-b) ** 2 + (4-3*a-b)**2

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot_surface(a, b, SSE, cmap='rainbow')
ax.contour(a, b, SSE, zdir='z', offset=0, cmap='rainbow')
plt.show()
