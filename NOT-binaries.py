# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:11:30 2024

@author: xGeeRe
"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Parameters for the binary systems
binaries = [
    {"name": "2006088484204609408", "period": 2.5960645*24, "t0": -16.87055556, "amplitude": 100},
    {"name": "2175699216614191360", "period": 4.769977*24, "t0": -85.41666667, "amplitude": 50},
    {"name": "2061252975440642816", "period": 2.139183*24, "t0":-29.39916667, "amplitude": 50}
]

# Observation times in hours since October 28, 5pm
observation_times = [
    [4.6, 6.5, 7.5, 8.5, 4.6+24, 6.5+24, 7.5+24, 8.5+24],      # Observation times (hours) for Binary 1
    [4.1, 6.1, 7, 4.1+24, 6.1+24, 7+24],         # Observation times (hours) for Binary 2
    [3.5, 5.5, 27.5, 29.5]          # Observation times (hours) for Binary 3
]

# Time grid for plotting curves from Hour 0 (Oct 28, 5pm) to Hour 39 (Oct 30, 8am)
time_grid = np.linspace(0, 39, 500)

# Start datetime for reference
start_datetime = datetime(2024, 10, 28, 17, 0)  # October 28, 5pm

# Function to convert hours to real datetime labels
def hours_to_datetime(hours):
    real_times = [start_datetime + timedelta(hours=int(h)) for h in hours]
    return [time.strftime("%H:%M") for time in real_times]


# Plot settings
plt.figure(figsize=(12, 8))
colors = ['b', 'g', 'r']  # Color for each binary's plot

for i, binary in enumerate(binaries):
    # Compute the RV curve
    rv_curve = binary["amplitude"] * np.sin(2 * np.pi * (time_grid - binary["t0"]) / binary["period"])
    
    # Plot the RV curve
    period_legend = binary['period']/24
    plt.plot(time_grid, rv_curve, label=fr"{binary['name']} (P={period_legend} d)", color=colors[i])
    
    # Plot observation points
    obs_times = observation_times[i]
    obs_rv = binary["amplitude"] * np.sin(2 * np.pi * (np.array(obs_times) - binary["t0"]) / binary["period"])
    plt.scatter(obs_times, obs_rv, color=colors[i], marker='o', edgecolor='k', zorder=5)
    # for j, t in enumerate(obs_times):
    #     plt.text(t, obs_rv[j], f"{t:.1f}h", fontsize=8, ha='right')

plt.axvline(2.8, ls='--', color='k')
plt.axvspan(-2, 2.8, alpha=0.3, color='k')
plt.axvline(13, ls='--', color='k')
plt.axvspan(13, 26.8, alpha=0.3, color='k')
plt.axvline(26.8, ls='--', color='k')
plt.axvline(37, ls='--', color='k')
plt.axvspan(37, 42, alpha=0.3, color='k')
# Labels and legend
plt.xlabel("Time (hours since Oct 28, 5:00 pm)", fontsize=16)
plt.ylabel("Radial Velocity (km/s)", fontsize=16)
plt.xlim(-2, 42)
# plt.title("Simulated Radial Velocity Curves with Observation Points (Hours)")

# Create a secondary x-axis on top for real date-time labels
ax = plt.gca()
secax = ax.secondary_xaxis('top')
secax.set_xticks(np.arange(0, 40, 6))  # Major ticks every 6 hours
secax.set_xticklabels(hours_to_datetime(np.arange(0, 40, 6)))
secax.set_xlabel("Real Time (HH:MM)", fontsize=16)
secax.set_xticks(np.arange(0, 40, 1))

plt.grid(True, which='both', axis='x')
plt.minorticks_on()
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

plt.legend()
plt.grid(True)
plt.show()


