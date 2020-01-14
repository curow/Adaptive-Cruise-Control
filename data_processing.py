# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = "./acc20200101-164816.csv"

# %%
df = pd.read_csv(filename, index_col=[0])

# %%
df.head()

# %%
df.tail()

# %%
df.describe()


# %%
df.plot(kind='line',x='timestamp',y='leader_vehicle_v',color='red')
df.plot(kind='line',x='timestamp',y='ego_vehicle_v',color='blue')
plt.show()

# %%
df.plot(kind='line',x='timestamp',y=['ego_vehicle_v', 'leader_vehicle_v'],color=['blue', 'red'])

# %%
skip = 1
ego_location = np.array([ 
    df['ego_vehicle_x'].values[::skip],
    df['ego_vehicle_y'].values[::skip],
    df['ego_vehicle_z'].values[::skip]
    ])

leader_location = np.array([ 
    df['leader_vehicle_x'].values[::skip],
    df['leader_vehicle_y'].values[::skip],
    df['leader_vehicle_z'].values[::skip]
    ])

# %%
distance_vectors = ego_location - leader_location
print(distance_vectors.shape)

# %%
distance = np.linalg.norm(distance_vectors, axis=0)
print(distance.shape)

# %%
timestamp = df['timestamp'].values
ego_vehicle_v = df['ego_vehicle_v'].values
leader_vehicle_v = df['leader_vehicle_v'].values
plt.plot(timestamp, distance)
plt.plot(timestamp, ego_vehicle_v)
plt.plot(timestamp, leader_vehicle_v)
plt.legend(['distance between the two (meters)', 'ego vehicle speed (kmh)', 'leader vehicle speed (kmh)'], loc='upper right')
plt.show()

# %%
plt.plot(timestamp, distance)
plt.legend(['distance between the two (meters)'], loc='upper right')
plt.show()

# %%
plt.plot(timestamp, ego_vehicle_v)
plt.plot(timestamp, leader_vehicle_v)
plt.legend(['ego vehicle speed (kmh)', 'leader vehicle speed (kmh)'], loc='upper right')
plt.show()
