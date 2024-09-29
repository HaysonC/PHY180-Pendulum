import math
import os
import pandas as pd  # Ensure pandas is imported
import matplotlib.pyplot as plt
import numpy as np

from data_analysis import *  # Assuming this is defined elsewhere

bin_size = 0.1

# Include all CSV files from the directory 'runs'
data = [
    data_analysis("runs/" + file)
    for file in os.listdir("runs")
    if file.endswith(".csv")
]
period_time_data = []
totalData = 0

# Get the total number of data points
for file in os.listdir("runs"):
    if file.endswith(".csv"):
        df = pd.read_csv("runs/" + file)
        totalData += len(df)
print(f"Total data: {totalData}")

for df, periods, anti_periods in data:
    max_angles = {
        i: df["theta"][df["period"] == i].max() for i in range(1, len(periods))
    }
    min_angles = {
        i: df["theta"][df["anti period"] == i].min()
        for i in range(1, len(anti_periods))
    }

    for i in max_angles.keys():
        if i not in [1, len(max_angles), 2, len(max_angles) - 1]:
            period_time_data.append({"Angle": max_angles[i], "Period Time": periods[i]})

    for i in min_angles.keys():
        if i not in [1, len(min_angles), 2, len(min_angles) - 1]:
            period_time_data.append(
                {"Angle": min_angles[i], "Period Time": anti_periods[i]}
            )

print(f"Total period data: {len(period_time_data)}")

# Create DataFrame from collected data
Period_Time = pd.DataFrame(period_time_data)
# Convert to radians
Period_Time["Angle"] = Period_Time["Angle"] * math.pi / 180

# Bin the data by angle in radians
min_value = Period_Time["Angle"].min()
max_value = Period_Time["Angle"].max()
bin_edges = np.arange(min_value, max_value + bin_size, bin_size)
Period_Time["Angle"] = pd.cut(Period_Time["Angle"], bins=bin_edges)

# Calculate the midpoint of each bin (in radians) and drop NaNs
Period_Time["Angle"] = Period_Time["Angle"].apply(
    lambda x: x.mid if pd.notnull(x) else np.nan
)
Period_Time = Period_Time.dropna()

# Convert back to float for calculations
Period_Time["Angle"] = Period_Time["Angle"].astype(float)

# Group and calculate mean values
Period_Time = Period_Time.groupby("Angle").mean().reset_index()

# Calculate error values
n = Period_Time["Period Time"].count()
yerr = Period_Time.groupby("Angle")["Period Time"].std().reset_index()[
    "Period Time"
] / np.sqrt(n)
xerr = bin_size / 4

# Fit a polynomial to the data
model = np.poly1d(np.polyfit(Period_Time["Angle"], Period_Time["Period Time"], 2))
a, b, c = model[2], model[1], model[0]


def fit_curve(t):
    return a * t**2 + b * t + c


# Calculate sigma
sigma = np.sqrt(
    np.sum((Period_Time["Period Time"] - fit_curve(Period_Time["Angle"])) ** 2)
    / (len(Period_Time) - 3)
)
# Calculate r_value
r_value = np.corrcoef(Period_Time["Period Time"], fit_curve(Period_Time["Angle"]))[0, 1]

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [2.4, 1]})
ax1.plot(Period_Time["Angle"], fit_curve(Period_Time["Angle"]), label="Regression Line")
ax1.scatter(
    Period_Time["Angle"],
    Period_Time["Period Time"],
    label="Data Points",
    s=5,
    color="red",
)
ax1.errorbar(
    Period_Time["Angle"],
    Period_Time["Period Time"],
    yerr=yerr,
    xerr=xerr,
    fmt="none",
    label="Error Bar",
    capsize=3,
    capthick=1,
    color="red",
)

ax1.set_xlabel("Angle (rad)")
ax1.set_ylabel("Period Time (s)")
ax1.legend()
ax1.set_title("Period Time vs Angle")
ax1.set_ylim(bottom=1.40, top=1.60)
plt.subplots_adjust(hspace=0.5)

# Calculate ±2 standard deviation values for residuals
upper_sigma = 2 * sigma
lower_sigma = -2 * sigma

# Plot residuals
ax2.scatter(
    Period_Time["Angle"],
    Period_Time["Period Time"] - fit_curve(Period_Time["Angle"]),
    label="Error",
    s=5,
    color="red",
)
ax2.axhline(y=0, color="black", linestyle="-")  # Residual = 0 line

# Calculate sigma for ±σ and ±2σ lines
upper_sigma = sigma
lower_sigma = -sigma
upper_2sigma = 2 * sigma
lower_2sigma = -2 * sigma

# Plot ±σ and ±2σ lines without labels
ax2.axhline(y=upper_sigma, color="grey", linestyle="--", alpha=0.5)
ax2.axhline(y=lower_sigma, color="grey", linestyle="--", alpha=0.5)
ax2.axhline(y=upper_2sigma, color="grey", linestyle="--", alpha=0.5)
ax2.axhline(y=lower_2sigma, color="grey", linestyle="--", alpha=0.5)

# Add text annotations for ±σ and ±2σ
sigma_x = 1.77
ax2.text(sigma_x, upper_sigma, r"$+\sigma$", ha="right", color="grey")
ax2.text(sigma_x, lower_sigma, r"$-\sigma$", ha="right", color="grey")
ax2.text(sigma_x, upper_2sigma, r"$+2\sigma$", ha="right", color="grey")
ax2.text(sigma_x, lower_2sigma, r"$-2\sigma$", ha="right", color="grey")

# Make more dense y-axis levels
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.errorbar(
    Period_Time["Angle"],
    Period_Time["Period Time"] - fit_curve(Period_Time["Angle"]),
    yerr=yerr,
    xerr=xerr,
    fmt="none",
    color="red",
    capsize=3,
    capthick=1,
)

# Set labels
ax2.set_xlabel("Angle (rad)")
ax2.set_ylabel("Residual (s)")

plt.savefig("graph1.png", dpi=300)
plt.show()

# Print n, sigma, r^2
print("n:", n)
print("sigma:", sigma)
print("r^2:", r_value**2)
print("a:", a)
print("b:", b)
print("c:", c)
