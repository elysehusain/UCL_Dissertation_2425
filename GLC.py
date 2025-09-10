# GLC data 

# %%
# Libraries
import numpy as np
import pandas as pd

# %% 
# Load data
glc_05_15 = pd.read_excel('All_data/glcnominalddata/GLC Nominal daily data_2005 to 2015.xlsx', sheet_name='4. spot curve', skiprows=3)
glc_16_24 = pd.read_excel('All_data/glcnominalddata/GLC Nominal daily data_2016 to 2024.xlsx', sheet_name='4. spot curve', skiprows=3)
glc_25_pres = pd.read_excel('All_data/glcnominalddata/GLC Nominal daily data_2025 to present.xlsx', sheet_name='4. spot curve', skiprows=3)

# %%
# Combine data
glc = pd.concat([glc_05_15, glc_16_24, glc_25_pres], ignore_index=True)

# Call 'years:' column 'Date'
glc = glc.rename(columns={'years:': 'Date'})

# Remove rows with NA in Date column
glc = glc[~glc['Date'].isna()]

# Rename 1 column to 1Y
glc = glc.rename(columns={1: "1Y"})

# Keep only Date and 1Y columns
glc = glc[['Date', '1Y']]


# %%
# Save cleaned data in Final_data folder
glc.to_csv('Final_data/GLC_1Y.csv', index=False)
# %%
