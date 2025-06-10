######
# M-bar Technologies and Consulting, LLC
# May 29, 2025
# OEIS Docket 2026-2028-Base-WMPs
#
# Prepared on behalf of MGRA
#
# License: GNU Public License v3
#
# Data has been analyzed from the SDG&E data request response Excel file:
# SDGE_Wildfire_Fatality_Structures_ratios_2025_05_12_0.xlsx, which was re-analyzed and stored in
# SDGE_Wildfire_Fatality_Structures_ratios_2025_05_12_0-jwm.xlsx. Modified calculations are on Tab SigRecent
#
# SDG&E original data on all fatal California wildfires was obtained from Wikipedia.
# MGRA analysis applied additional filters:
#  - only wildfires 1990 and later
#  - more than 10 structures destroyed
#  This left 30 of the 43 wildfires included in the SDG&E sample.
#  The average number of fatalities per structure was reduced from .0618 (SDG&E WMP) to 0.438 (MGRA)
#
#  A statistical analysis was prepared via prompts to ChatGPT AI, which authored portions of this code.
#  All AI results and claims were analyzed and validated. 
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Data
data = {
    "Log_Structures": [2.971, 3.124, 2.447, 2.804, 2.346, 2.932, 3.173, 3.371, 3.027, 3.45, 2.049, 3.208,
                       1.699, 3.001, 2.086, 3.217, 4.274, 3.216, 3.001, 2.739, 3.173, 3.291, 2.985, 3.182,
                       3.132, 2.893, 3.752, 2.736, 2.98, 3.462],
    "Deaths_Per_Structure": [0.001069519, 0.000752445, 0.003571429, 0, 0, 0, 0.004024145, 0.006377551,
                             0.021636877, 0.005319149, 0, 0.004956629, 0, 0.000997009, 0, 0.001212121,
                             0.004520315, 0.001825928, 0.005982054, 0.01459854, 0.000671141, 0.002046036,
                             0.002072539, 0, 0.002214022, 0.007682458, 0.003898635, 0.016544118, 0.001048218,
                             0.00862069]
}

df = pd.DataFrame(data)

# Compute Structures and RMS
df["Structures"] = 10 ** df["Log_Structures"]
df["RMS"] = np.where(df["Deaths_Per_Structure"] > 0,
                     np.sqrt(df["Deaths_Per_Structure"] / df["Structures"]),
                     0)

valid_data = df[df["Deaths_Per_Structure"] > 0]

# Hypothesis line
#x_vals = np.array(df["Log_Structures"])
x_vals = valid_data["Log_Structures"]
y_obs = valid_data["Deaths_Per_Structure"]
y_obs_v = y_obs.values
y_mean = np.mean(y_obs)
n_obs = len(y_obs)
sigma = valid_data["RMS"].values
sdge_avg = .00618

sdge_line = np.full_like(y_obs, sdge_avg)
sdge_draw = np.full_like(df["Log_Structures"],sdge_avg)

# MGRA alternative hypothesis
mgra_avg = 0.00438
mgra_line = np.full_like(y_obs, mgra_avg)
mgra_draw = np.full_like(df["Log_Structures"],mgra_avg)


ss_model = ((y_mean - sdge_line) ** 2) * n_obs
ss_resid = np.sum((y_obs - sdge_avg) ** 2)
ss_total = np.sum((y_obs - y_mean) ** 2)
r_squared = 1 - (ss_resid / ss_total)

ss_model_mgra = ((y_mean - mgra_line) ** 2) * n_obs
ss_resid_mgra = np.sum((y_obs - mgra_avg) ** 2)
ss_total_mgra = np.sum((y_obs - y_mean) ** 2)
r_squared_mgra = 1 - (ss_resid_mgra / ss_total_mgra)

# Degrees of Freedom
df_model = 1 #linear fit
df_resid = n_obs - 1

# Mean squares
# ms_model = (ss_total - ss_resid) / df_model
ms_model = ss_resid / df_resid
ms_resid = ss_resid / df_resid

# F-stat
f_stat = ms_model / ms_resid
p_value = 1 - stats.f.cdf(f_stat, df_model, df_resid)

# Chi-squared SDG&E
chi_sq = np.sum(((y_obs_v - sdge_avg) / sigma) ** 2)
chi_sq_red = chi_sq / df_resid
p_value_chi2 = 1 - stats.chi2.cdf(chi_sq, df_resid)

# Chi-squared MGRA
chi_sq_mgra = np.sum(((y_obs_v - mgra_avg) / sigma) ** 2)
chi_sq_red_mgra = chi_sq_mgra / df_resid
p_value_chi2_mgra = 1 - stats.chi2.cdf(chi_sq_mgra, df_resid)


# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(
    df["Log_Structures"],
    df["Deaths_Per_Structure"],
    yerr=df["RMS"],
    fmt='o',
    color='blue',
    label='California Wildfires',
    capsize=3
)
plt.plot(df["Log_Structures"], sdge_draw, color='red', linestyle='-', label='SDG&E average')
plt.plot(df["Log_Structures"], mgra_draw, color='green', linestyle='-', label='MGRA average')

plt.xlabel('Log10(Structures)')
plt.ylabel('Deaths per Structure')
plt.title('Fatalities per Structure vs. Structures (log10)')
plt.legend()
plt.grid(True)

# Add R² and P-value in a box
textstr = f'SDG&E Avg = {sdge_avg:.5f}\nReduced $\\chi$² = {chi_sq_red:.3f}\nP-value = {p_value_chi2:.3f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)
# Add R² and P-value in a box - MGRA
textstr_mgra = f'MGRA Avg = {mgra_avg:.5f}\nReduced $\\chi$² = {chi_sq_red_mgra:.3f}\nP-value = {p_value_chi2_mgra:.3f}'
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)
plt.text(0.05, 0.65, textstr_mgra, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
