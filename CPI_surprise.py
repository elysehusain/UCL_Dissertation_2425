# CPI Surprise - Validation 

# %%
# Libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %% 
# (1) Load Data

# CPI data
cpi = pd.read_excel('All_data/CPI/Economic Indicator_United Kingdom CPI YY_28 Aug 2025.xlsx', sheet_name='First Release Data', skiprows=3)

# Reuters Forecast data
reuters = pd.read_csv('All_data/Reuters/reuters_forecast.csv')

# ILS data
ils = pd.read_csv("Final_data/GBP_inflation_swaps.csv")

# %%
# 2 year ILS
ils_2y = pd.read_excel('All_data/Swaps/2Y/Price History_20250907_1540.xlsx', skiprows=17)

# Subtract Bid and Ask columns to get Midpoint
ils_2y["Midpoint"] = (ils_2y["Bid"] + ils_2y["Ask"]) / 2

# Keep only Date and Midpoint columns
ils_2y = ils_2y[["Date", "Midpoint"]]

# Rename Midpoint to ILS_2Y
ils_2y = ils_2y.rename(columns={"Midpoint": "ILS_2Y"})

# Coerce Date to datetime and make midnight
ils_2y["Date"] = pd.to_datetime(ils_2y["Date"], errors="coerce")
ils_2y["Date"] = ils_2y["Date"].dt.normalize()

# Sort and set index
ils_2y = ils_2y.sort_values("Date").set_index("Date")


# %%
# (2) Data Cleaning

# (i) CPI data

# Coerce Original Release Date to datetime and make 07:00 to midnight
cpi["Original Release Date"] = pd.to_datetime(cpi["Original Release Date"], dayfirst=True, errors="coerce")
cpi["Original Release Date"] = cpi["Original Release Date"].dt.normalize()

# Rename First Release to CPI
cpi = cpi.rename(columns={"First Release": "CPI"})
cpi["CPI"] = pd.to_numeric(cpi["CPI"], errors="coerce")

# Rename Original Release Date to Date
cpi = cpi.rename(columns={"Original Release Date": "Date"})

# Only keep 'Original Release Date' and 'CPI'
cpi = cpi[["Date", "CPI"]]

# (ii) Reuters Forecast data

# Coerce Release Date to datetime and make midnight
reuters["Date"] = pd.to_datetime(reuters["Date"], dayfirst=True, errors="coerce")
reuters["Date"] = reuters["Date"].dt.normalize()

# Rename Confidence to reuters_forecast
reuters = reuters.rename(columns={"Confidence": "reuters_forecast"})

# (iii) ILS data

# Coerce Date to datetime and make midnight
ils["Date"] = pd.to_datetime(ils["Date"], errors="coerce")
ils["Date"] = ils["Date"].dt.normalize()

# Rename Midpoint to ILS
ils = ils.rename(columns={"Midpoint": "ILS"})

# Keep only Date and ILS
ils = ils[["Date", "ILS"]]

# Sort and set index
ils = ils.sort_values("Date").set_index("Date")

# %%
# (3) Merge CPI and Reuters Data

# Make reference months for CPI and Reuters so the correct forecasts are matched
cpi["ref_month"] = cpi["Date"].dt.to_period("M")
reuters["ref_month"] = reuters["Date"].dt.to_period("M")

# Merge on ref_month
merged = pd.merge(cpi, reuters, on="ref_month", suffixes=("_cpi", "_reuters"))

# Surprise
merged["surprise"] = merged["CPI"] - merged["reuters_forecast"]

# %%
# (4) Merge with ILS Data

# Function to get ILS change on release date
def ils_change_on_release(d, series, col):
    try:
        return series.loc[d, col] - series.loc[d - pd.Timedelta(days=1), col]
    except KeyError:
        return None

# 1Y ILS change
merged["ILS_1Y_change"] = merged["Date_cpi"].apply(lambda d: ils_change_on_release(d, ils, "ILS"))

# 2Y ILS change
merged["ILS_2Y_change"] = merged["Date_cpi"].apply(lambda d: ils_change_on_release(d, ils_2y, "ILS_2Y"))

# Drop rows with missing surprises or ILS changes
merged = merged.dropna(subset=["surprise", "ILS_1Y_change", "ILS_2Y_change"])

# Remove outliers beyond 3 standard deviations
threshold_1y = 3 * merged["ILS_1Y_change"].std()
merged = merged[(merged["ILS_1Y_change"].abs() <= threshold_1y)]
print(f"Removed {len(merged) - len(merged2)} outliers based on 1Y ILS changes")

# %%
# (5) Regression Analysis

# --- 1Y ILS ---
X1 = sm.add_constant(merged["surprise"])
y1 = merged["ILS_1Y_change"]
model1 = sm.OLS(y1, X1).fit()
print("=== 1Y ILS ===")
print(model1.summary())

# --- 2Y ILS ---
X2 = sm.add_constant(merged["surprise"])
y2 = merged["ILS_2Y_change"]
model2 = sm.OLS(y2, X2).fit()
print("\n=== 2Y ILS ===")
print(model2.summary())

# %%
# (6) Scatter Plots

fig, axes = plt.subplots(1, 2, figsize=(12,5))

# 1Y plot
axes[0].scatter(merged["surprise"], merged["ILS_1Y_change"], color="blue")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].axvline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("CPI Surprise (Actual - Forecast)")
axes[0].set_ylabel("ΔILS 1Y (t - t-1)")
axes[0].set_title("1Y ILS Reaction to CPI Surprises")

slope1 = model1.params["surprise"]
intercept1 = model1.params["const"]
axes[0].plot(merged["surprise"], intercept1 + slope1 * merged["surprise"], color="red")

# 2Y plot
axes[1].scatter(merged["surprise"], merged["ILS_2Y_change"], color="green")
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].axvline(0, color="black", linewidth=0.8)
axes[1].set_xlabel("CPI Surprise (Actual - Forecast)")
axes[1].set_ylabel("ΔILS 2Y (t - t-1)")
axes[1].set_title("2Y ILS Reaction to CPI Surprises")

slope2 = model2.params["surprise"]
intercept2 = model2.params["const"]
axes[1].plot(merged["surprise"], intercept2 + slope2 * merged["surprise"], color="red")

plt.tight_layout()
plt.show()


# %%
# (7) Event-Day vs Non-Event-Day Volatility Check

# Mark CPI release days
ils_all = ils.copy()
ils_all["ILS_1Y_change"] = ils_all["ILS"].diff()

ils2_all = ils_2y.copy()
ils2_all["ILS_2Y_change"] = ils2_all["ILS_2Y"].diff()

# Flag release days
release_dates = merged["Date_cpi"].unique()
ils_all["is_cpi_day"] = ils_all.index.isin(release_dates)
ils2_all["is_cpi_day"] = ils2_all.index.isin(release_dates)

# Calculate threshold (3 standard deviations)
threshold = 3 * ils_all["ILS_1Y_change"].std()
print(f"Threshold: {threshold * 10000:.1f} basis points")

# Remove outliers
ils_clean = ils_all[ils_all["ILS_1Y_change"].abs() <= threshold].copy()
print(f"Removed {len(ils_all) - len(ils_clean)} outliers")

# Compute mean absolute change on release vs non-release days
summary = pd.DataFrame({
    "ILS_1Y_abs_move": ils_all.groupby("is_cpi_day")["ILS_1Y_change"].apply(lambda x: x.abs().mean()),
    "ILS_2Y_abs_move": ils2_all.groupby("is_cpi_day")["ILS_2Y_change"].apply(lambda x: x.abs().mean())
})

print("Average absolute daily move (in % points):")
print(summary)

# %%
# (8) T-test: Is the volatility difference significant?

from scipy import stats

# Split ILS changes into CPI vs non-CPI days
cpi_days_1y = ils_all.loc[ils_all["is_cpi_day"], "ILS_1Y_change"].abs().dropna()
non_cpi_days_1y = ils_all.loc[~ils_all["is_cpi_day"], "ILS_1Y_change"].abs().dropna()

cpi_days_2y = ils2_all.loc[ils2_all["is_cpi_day"], "ILS_2Y_change"].abs().dropna()
non_cpi_days_2y = ils2_all.loc[~ils2_all["is_cpi_day"], "ILS_2Y_change"].abs().dropna()

# Run independent-samples t-tests
tstat_1y, pval_1y = stats.ttest_ind(cpi_days_1y, non_cpi_days_1y, equal_var=False)
tstat_2y, pval_2y = stats.ttest_ind(cpi_days_2y, non_cpi_days_2y, equal_var=False)

print("1Y ILS: t = %.3f, p = %.3f" % (tstat_1y, pval_1y))
print("2Y ILS: t = %.3f, p = %.3f" % (tstat_2y, pval_2y))

# %%
# (9) 1-Year ILS Validation Table - Key Statistics Only

# Calculate key statistics
cpi_days_1y = ils_all.loc[ils_all["is_cpi_day"], "ILS_1Y_change"].abs().dropna()
non_cpi_days_1y = ils_all.loc[~ils_all["is_cpi_day"], "ILS_1Y_change"].abs().dropna()

# Calculate percentage increase in volatility
pct_increase = ((cpi_days_1y.mean() - non_cpi_days_1y.mean()) / non_cpi_days_1y.mean()) * 100

# Create simple validation table
validation_table = pd.DataFrame({
    'Metric': [
        'Mean Absolute Change - CPI Days (percentage points)',
        'Mean Absolute Change - Non-CPI Days (percentage points)',
        'Percentage Increase on CPI Days',
        'Number of CPI Release Days',
        'Number of Non-CPI Days',
        't-statistic',
        'p-value',
        'Statistically Significant (5% level)'
    ],
    'Value': [
        f"{cpi_days_1y.mean() * 100:.2f}",
        f"{non_cpi_days_1y.mean() * 100:.2f}",
        f"{pct_increase:.1f}%",
        f"{len(cpi_days_1y)}",
        f"{len(non_cpi_days_1y)}",
        f"{tstat_1y:.3f}",
        f"{pval_1y:.3f}",
        "Yes" if pval_1y < 0.05 else "No"
    ]
})

print("Table 1: ILS Validation - Volatility on CPI Release Days")
print("=" * 60)
print(validation_table.to_string(index=False))

# Export to Excel
validation_table.to_excel('Final_data/ils_validation_table.xlsx', index=False, sheet_name='ILS_Validation')
print(f"\nTable exported to 'ils_validation_table.xlsx'")

# %%

# (10) Save Final Data

# Save final merged data for use in event study
merged.to_csv("Final_data/CPI_surprise.csv", index=False)

# %%
