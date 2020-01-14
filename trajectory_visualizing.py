import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update_lines(num, data_lines, lines):
    for line, data in zip(lines, data_lines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Reading trajectory from CSV file
filename = "./acc20200101-164816.csv"
df = pd.read_csv(filename, index_col=[0])
skip = 5
ego_data = np.array([ 
    df['ego_vehicle_x'].values[::skip],
    df['ego_vehicle_y'].values[::skip],
    df['ego_vehicle_z'].values[::skip]
    ])

leader_data = np.array([ 
    df['leader_vehicle_x'].values[::skip],
    df['leader_vehicle_y'].values[::skip],
    df['leader_vehicle_z'].values[::skip]
    ])
# shift leader data z axis by 3
leader_data[2, :] += 3

data = [ego_data, leader_data]

length = ego_data.shape[1]

# create trajectory line
ego_line = ax.plot(ego_data[0, 0:1], ego_data[1, 0:1], ego_data[2, 0:1])[0]
leader_line = ax.plot(leader_data[0, 0:1], leader_data[1, 0:1], leader_data[2, 0:1])[0]
lines = [ego_line, leader_line]

# Setting the axes properties
ax.set_xlim3d([-600, 400])
ax.set_xlabel('X')

ax.set_ylim3d([0, 250])
ax.set_ylabel('Y')

ax.set_zlim3d([-1, 12])
ax.set_zlabel('Z')

ax.set_title('3D trajectory')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, length, fargs=(data, lines), interval=1, blit=False)

plt.show()
