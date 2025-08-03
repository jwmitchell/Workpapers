import numpy as np
import pandas as pd
from scipy.stats import chisquare, chi2_contingency
import matplotlib.pyplot as plt

# --- Step 1: Input data ---
categories = [
    "Animal Contact", "Balloon Contact", "Vehicle Contact", "Vegetation Contact",
    "Other Contact*", "Conductor", "Equipment – Non-Conductor**",
    "Other All***", "Undetermined****", "Overhead to Underground Connection"
]

observed1 = np.array([20, 27, 20, 72, 47, 123, 412, 151, 10, 20])  # Sample 1
observed2 = np.array([19, 9, 10, 11, 4, 10, 49, 9, 1, 0])           # Sample 2

# --- Step 2: Compute expected values under pooled distribution ---
total1 = observed1.sum()
total2 = observed2.sum()
combined = observed1 + observed2
expected1 = combined * (total1 / (total1 + total2))
expected2 = combined * (total2 / (total1 + total2))

# --- Step 3: One-sample chi-squared tests ---
chi2_stat1, p_val1 = chisquare(f_obs=observed1, f_exp=expected1)
chi2_stat2, p_val2 = chisquare(f_obs=observed2, f_exp=expected2)

# --- Step 4: Two-sample chi-squared test (contingency) ---
contingency_table = np.array([observed1, observed2])
chi2_stat_total, p_value_total, dof, expected_table = chi2_contingency(contingency_table)

# --- Step 5: Table with contributions ---
df = pd.DataFrame({
    "Category": categories,
    "Sample 1": observed1,
    "Sample 2": observed2,
    "Combined": combined,
    "Expected Sample 1": expected1,
    "Expected Sample 2": expected2
})
df["Chi2 Contribution Sample 1"] = (observed1 - expected1) ** 2 / expected1
df["Chi2 Contribution Sample 2"] = (observed2 - expected2) ** 2 / expected2

# --- Step 6: Print outputs ---
print("Chi-Squared Test: Sample 1")
print(f"  χ² = {chi2_stat1:.2f}, p = {p_val1:.4g}")

print("Chi-Squared Test: Sample 2")
print(f"  χ² = {chi2_stat2:.2f}, p = {p_val2:.4g}")

print("Chi-Squared Test: Both Samples (Contingency Table)")
print(f"  χ² = {chi2_stat_total:.2f}, p = {p_value_total:.4g}, df = {dof}")

