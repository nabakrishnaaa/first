import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
try:
    df = pd.read_csv('result.csv')
except FileNotFoundError:
    print("Error: result.csv not found. Please run the C++ simulation first.")
    exit()

# 2. Setup the Plot
plt.figure(figsize=(10, 6))

# Plot AdamW Loss (Red)
plt.plot(df['step'], df['aloss'], color='#e74c3c', label='AdamW (Standard)', linewidth=2)

# Plot SOAP Loss (Blue)
plt.plot(df['step'], df['sloss'], color='#3498db', label='SOAP (Spectral Preconditioning)', linewidth=2)

# 3. Styling for Analysis
plt.yscale('log')  # Use Log Scale to see the fine details near convergence
plt.title("Convergence Comparison: AdamW vs SOAP", fontsize=16)
plt.xlabel("Optimization Steps", fontsize=12)
plt.ylabel("Loss Value (Log Scale)", fontsize=12)

# Add a horizontal line for the "Global Minimum" target
plt.axhline(y=1e-6, color='gray', linestyle='--', alpha=0.5, label='Target Loss')

plt.grid(True, which="both", ls="-", alpha=0.15)
plt.legend(fontsize=11)

# Annotate the "SOAP Advantage"
plt.annotate('SOAP converges faster\ndue to Eigen-alignment', 
             xy=(50, df['sloss'][50]), xytext=(80, 100),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
plt.savefig('convergence_line_graph.png', dpi=300)
plt.show()