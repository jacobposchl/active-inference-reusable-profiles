import numpy as np

print(f"================ Loading HEATMAP A ================")
# Load the .npz file
data = np.load('results/data/k2_vs_k1_grid_results_A.npz')

# List all arrays stored in the file
print(data.files)

# Example: print the log likelihood delta heatmap
print("Δ Log Likelihood (K=2 - K=1):")
print(data['ll_delta'])

# Example: print the accuracy delta heatmap
print("Δ Accuracy (K=2 - K=1):")
print(data['acc_delta'])

# Example: print the BIC delta heatmap
print("Δ BIC (K=2 - K=1):")
print(data['bic_delta'])

print(f"================ Loading HEATMAP B ================")
# Load the .npz file
data = np.load('results/data/k2_vs_k1_grid_results_B.npz')

# List all arrays stored in the file
print(data.files)

# Example: print the log likelihood delta heatmap
print("Δ Log Likelihood (K=2 - K=1):")
print(data['ll_delta'])

# Example: print the accuracy delta heatmap
print("Δ Accuracy (K=2 - K=1):")
print(data['acc_delta'])

# Example: print the BIC delta heatmap
print("Δ BIC (K=2 - K=1):")
print(data['bic_delta'])

print(f"================ Loading HEATMAP C ================")
# Load the .npz file
data = np.load('results/data/k2_vs_k1_grid_results_C.npz')

# List all arrays stored in the file
print(data.files)

# Example: print the log likelihood delta heatmap
print("Δ Log Likelihood (K=2 - K=1):")
print(data['ll_delta'])

# Example: print the accuracy delta heatmap
print("Δ Accuracy (K=2 - K=1):")
print(data['acc_delta'])

# Example: print the BIC delta heatmap
print("Δ BIC (K=2 - K=1):")
print(data['bic_delta'])
