import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

# 1. Load data from CSV
df = pd.read_csv('result.csv')

# 2. Setup the Rosenbrock surface for the background
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

x_range = np.linspace(-2, 2, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = rosenbrock(X, Y)

# 3. Create 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot surface (LogNorm makes the valley clear)
ax.plot_surface(X, Y, Z, cmap='viridis', norm=LogNorm(), alpha=0.3, antialiased=True)

# Plot AdamW Path (Red)
ax.plot(df['ax'], df['ay'], df['aloss'], color='red', label='AdamW Path', linewidth=2, zorder=10)
# Plot SOAP Path (Blue)
ax.plot(df['sx'], df['sy'], df['sloss'], color='blue', label='SOAP Path', linewidth=2, zorder=11)

# Highlight Minimum and Start
ax.scatter(1, 1, 0, color='gold', s=200, marker='*', label='Global Minimum')
ax.scatter(df['ax'].iloc[0], df['ay'].iloc[0], df['aloss'].iloc[0], color='black', s=50, label='Start')

# Axis Labels
ax.set_title("3D Comparison: AdamW vs SOAP (Spectral Preconditioning)", fontsize=15)
ax.set_xlabel("Parameter X")
ax.set_ylabel("Parameter Y")
ax.set_zlabel("Loss")
ax.view_init(elev=35, azim=-120)
plt.legend()
plt.show()