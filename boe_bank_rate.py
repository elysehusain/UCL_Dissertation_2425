# Bank Rate Decisions file 

# %% 

# Libraries
import pandas as pd

# %%
# Read data
bank_rate_data = pd.read_excel('All_data/mpcvoting.xlsx', sheet_name='Bank Rate Decisions')

# %%
# Split into two dataframes: one for bank rate decisions, one for voting summary

# (i) Clean up the bank rate decisions data

# Copy as a new dataframe
bank_rate_decisions = bank_rate_data.copy()

# Row 2 contains column title info, so make these the headers
bank_rate_decisions.columns = bank_rate_decisions.iloc[2]

# Remove first column 
bank_rate_decisions = bank_rate_decisions.iloc[:,1:]

# Rename column 0 as "Date"
bank_rate_decisions.rename(columns={bank_rate_decisions.columns[0]: 'date'}, inplace=True)

# Convert the 'Date' column to datetime format
bank_rate_decisions['date'] = pd.to_datetime(bank_rate_decisions['date'], errors='coerce')

# Rename column 1
bank_rate_decisions.rename(columns={bank_rate_decisions.columns[1]: 'new_bank_rate'}, inplace=True)

# Create previous bank rate column
bank_rate_decisions['previous_bank_rate'] = bank_rate_decisions['new_bank_rate'].shift(1)

# Move column to column 2
bank_rate_decisions = bank_rate_decisions[['date', 'new_bank_rate', 'previous_bank_rate'] + list(bank_rate_decisions.columns[2:])]

# Keep dates after 2015
bank_rate_decisions = bank_rate_decisions[bank_rate_decisions['date'] > '2015-01-01']

# Remove columns that contain only NaN values
bank_rate_decisions = bank_rate_decisions.dropna(axis=1, how='all')

# Reset the index
bank_rate_decisions.reset_index(drop=True, inplace=True)

# 93 rows = 93 bank rate decisions = 93 minutes 

# %%

# (ii) Clean up the voting summary data

# Copy as a new dataframe
voting_summary = bank_rate_data.copy()

# Row 2 contains column title info, so make these the headers
voting_summary.columns = voting_summary.iloc[2]

# Select columns & rows
voting_summary = voting_summary.iloc[3:8, 1:]

# Remove columns that contain only NaN values
voting_summary = voting_summary.dropna(axis=1, how='all')

# Rename first column as 'Vote'
voting_summary.rename(columns={voting_summary.columns[0]: 'Vote'}, inplace=True)

# Swap columns and rows
voting_summary = voting_summary.transpose()

# Row 1 contains column title info, so make these the headers
voting_summary.columns = voting_summary.iloc[0]

# Index is currently the first column, so reset the index
voting_summary.reset_index(drop=False, inplace=True)

# Rename column 0 as 'mpc_member'
voting_summary.rename(columns={voting_summary.columns[0]: 'mpc_member'}, inplace=True)

# Remove row 0
voting_summary = voting_summary.iloc[1:]

# Reset the index
voting_summary.reset_index(drop=True, inplace=True)

# Remove column titled 'nan'
voting_summary = voting_summary.loc[:, voting_summary.columns.notnull()]

# Create surname column 
voting_summary['Surname'] = voting_summary['mpc_member'].str.split().str[-1]

# Create first name column 
voting_summary['First_Name'] = voting_summary['mpc_member'].str.split().str[0]

# %%

# Read in MPC members data
mpc_members = pd.read_csv('All_data/cleaned_mpc_members.csv')

# Merge voting summary with MPC members data on first and surname
mpc_df = pd.merge(mpc_members, voting_summary, on=['First_Name', 'Surname'], how='left')

# %%

# Save the cleaned dataframes to CSV files

# Save bank rate decisions
bank_rate_decisions.to_csv('Final_data/cleaned_bank_rate_decisions.csv', index=False)

# Save mpc voting summary
mpc_df.to_csv('Final_data/cleaned_mpc_voting_summary.csv', index=False)

# %%
