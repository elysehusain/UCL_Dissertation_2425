# Cleaning swaps data from Refinitiv Workspace

# %%
# Libraries
import pandas as pd

# %%
# Read in data
swaps_df = pd.read_excel("All_data/Swaps/Price History_20250808_1531.xlsx", skiprows=17, header=0)

# Remove na columns
swaps_df = swaps_df.dropna(axis=1, how="all")

# Cread midpoint column
swaps_df["Midpoint"] = ((swaps_df["Bid"] + swaps_df["Ask"]) / 2).round(4)

# %%
# Save to csv 
swaps_df.to_csv("Final_data/GBP_inflation_swaps.csv", index=False)
