import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load correlation matrix
file_path = "correlation_matrix.csv"  # Update with actual path
df_corr = pd.read_csv(file_path, index_col=0)

# Set correlation threshold (e.g., 0.3)
corr_threshold = 0.2

# Mask weak correlations
filtered_corr = df_corr[(df_corr >= corr_threshold) | (df_corr <= -corr_threshold)]

# Plot heatmap
plt.figure(figsize=(15, 8))
sns.heatmap(filtered_corr, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Filtered Correlation Heatmap")
plt.show()
