import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    "SOC (%)": [43.7, 95.6, 30.5, 78.1, 60.2],
    "Voltage (V)": [3.62, 3.87, 3.55, 3.78, 3.60],
    "Current (A)": [33.5, 32.2, 31.1, 34.0, 33.0],
    "Battery Temp (C)": [25, 30, 27, 26, 29],
    "Ambient Temp (C)": [22, 35, 30, 28, 26],
    "Charging Duration (min)": [30, 45, 20, 35, 40],
    "Degradation Rate (%)": [0.5, 0.7, 0.6, 0.4, 0.5],
    "Efficiency (%)": [90, 85, 88, 89, 87],
    "Charging Cycles": [112, 398, 210, 300, 150],
})

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

sns.kdeplot(data["SOC (%)"], ax=axes[0], label="Original", fill=True)
sns.kdeplot(scaled_df["SOC (%)"], ax=axes[1], label="Scaled", fill=True)
axes[0].set_title("SOC (%) Before Scaling")
axes[1].set_title("SOC (%) After Scaling")

plt.tight_layout()
plt.savefig("feature_scaling_soc.png")
plt.show()
