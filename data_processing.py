import pandas as pd
import matplotlib.pyplot as plt
import time
import random

df = pd.from_csv('./out/acc.csv')
 
ysample1 = random.sample(range(-50, 50), 100)
ysample2 = random.sample(range(-50, 50), 100)
 
xdata = []
ydata1 = []
ydata2 = []
 
plt.show()
 
axes = plt.gca()
axes.set_xlim(0, 100)
axes.set_ylim(-50, +50)
line1, = axes.plot(xdata, ydata1, 'r-')
line2, = axes.plot(xdata, ydata2, 'g-')
 
for i in range(100):
    xdata.append(i)
    ydata1.append(ysample1[i])
    line1.set_xdata(xdata)
    line1.set_ydata(ydata1)
    ydata2.append(ysample2[i])
    line2.set_xdata(xdata)
    line2.set_ydata(ydata2)
    plt.draw()
    plt.pause(1e-17)
 
# add this if you don't want the window to disappear at the end
plt.show()