# Event study for BoE Communication and Inflation Expectations

# %%
# Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# %%

########### (1) Load Data ###########

# Read sentiment data 
sentiment_data = pd.read_csv('Text Data - BoE/Final_data/processed_events_with_lm_ecb_finbert_fomc_sentiment.csv', parse_dates=['event_date'])
growth_events_data = pd.read_csv('Text Data - BoE/Final_data/growth_events_with_lm_ecb_finbert_fomc_sentiment.csv', parse_dates=['event_date'])
inflation_events_data = pd.read_csv('Text Data - BoE/Final_data/inflation_events_with_lm_ecb_finbert_fomc_sentiment.csv', parse_dates=['event_date'])

# Read inflation expectations data
ils_data = pd.read_csv('Inflation Expectations Data/Final_data/GBP_inflation_swaps.csv', parse_dates=['Date'])

# Read GLC data
glc = pd.read_csv('Inflation Expectations Data/Final_data/GLC_1Y.csv')

# CPI data
cpi_surprise = pd.read_csv('Inflation Expectations Data/Final_data/CPI_surprise.csv')


# %%

########### (2) Data Cleaning ###########

######################
# (i) Sentiment data
######################

# Normalise all sentiment scores
sentiment_data['norm_lm_sent'] = (sentiment_data['net_lm_sent'] - sentiment_data['net_lm_sent'].mean()) / sentiment_data['net_lm_sent'].std()
sentiment_data['norm_ecb_sent'] = (sentiment_data['net_ecb_sentiment'] - sentiment_data['net_ecb_sentiment'].mean()) / sentiment_data['net_ecb_sentiment'].std()
sentiment_data['norm_finbert_sent'] = (sentiment_data['finbert_sentiment'] - sentiment_data['finbert_sentiment'].mean()) / sentiment_data['finbert_sentiment'].std()
sentiment_data['norm_fomc_sent'] = (sentiment_data['fomc_roberta_sentiment'] - sentiment_data['fomc_roberta_sentiment'].mean()) / sentiment_data['fomc_roberta_sentiment'].std()

# Create new df 'norm_sentiment' with normalised sentiment scores
norm_sentiment = sentiment_data[['event_date', 'event_type', 'filename', 'speaker', 'is_mpc_member', 'norm_lm_sent', 'norm_ecb_sent', 'norm_finbert_sent', 'norm_fomc_sent']]

# Remove rows where event_type is minutes and date is before 2015-08-01
norm_sentiment = norm_sentiment[~((norm_sentiment['event_type'] == 'minutes') & (norm_sentiment['event_date'] < '2015-08-01'))].reset_index(drop=True)

# Remove rows with NA in 'growth_sentences'
growth_events_data = growth_events_data.dropna(subset=['growth_sentences']).reset_index(drop=True)

# Normalise sentiment scores in growth and inflation events data
growth_events_data['norm_lm_sent'] = (growth_events_data['growth_sentiment_lm'] - growth_events_data['growth_sentiment_lm'].mean()) / growth_events_data['growth_sentiment_lm'].std()
growth_events_data['norm_ecb_sent'] = (growth_events_data['growth_sentiment_ecb'] - growth_events_data['growth_sentiment_ecb'].mean()) / growth_events_data['net_lm_sent'].std()
growth_events_data['norm_finbert_sent'] = (growth_events_data['growth_sentiment_finbert'] - growth_events_data['growth_sentiment_finbert'].mean()) / growth_events_data['growth_sentiment_finbert'].std()
growth_events_data['norm_fomc_sent'] = (growth_events_data['growth_sentiment_fomc'] - growth_events_data['growth_sentiment_fomc'].mean()) / growth_events_data['growth_sentiment_fomc'].std()

# Create new df 'norm_growth_sentiment' with normalised sentiment scores
norm_growth_sentiment = growth_events_data[['event_date', 'event_type', 'filename', 'speaker', 'is_mpc_member', 'norm_lm_sent', 'norm_ecb_sent', 'norm_finbert_sent', 'norm_fomc_sent']]

# Remove rows where event_type is minutes and date is before 2015-08-01
norm_growth_sentiment = norm_growth_sentiment[~((norm_growth_sentiment['event_type'] == 'minutes') & (norm_growth_sentiment['event_date'] < '2015-08-01'))].reset_index(drop=True)

# Remove rows with NA in 'inflation_sentences'
inflation_events_data = inflation_events_data.dropna(subset=['inflation_sentences']).reset_index(drop=True)

inflation_events_data['norm_lm_sent'] = (inflation_events_data['inflation_sentiment_lm'] - inflation_events_data['inflation_sentiment_lm'].mean()) / inflation_events_data['inflation_sentiment_lm'].std()
inflation_events_data['norm_ecb_sent'] = (inflation_events_data['inflation_sentiment_ecb'] - inflation_events_data['inflation_sentiment_ecb'].mean()) / inflation_events_data['inflation_sentiment_ecb'].std()
inflation_events_data['norm_finbert_sent'] = (inflation_events_data['inflation_sentiment_finbert'] - inflation_events_data['inflation_sentiment_finbert'].mean()) / inflation_events_data['inflation_sentiment_finbert'].std()
inflation_events_data['norm_fomc_sent'] = (inflation_events_data['inflation_sentiment_fomc'] - inflation_events_data['inflation_sentiment_fomc'].mean()) / inflation_events_data['inflation_sentiment_fomc'].std()

# Create new df 'norm_inflation_sentiment' with normalised sentiment scores
norm_inflation_sentiment = inflation_events_data[['event_date', 'event_type', 'filename', 'speaker', 'is_mpc_member', 'norm_lm_sent', 'norm_ecb_sent', 'norm_finbert_sent', 'norm_fomc_sent']]

# Remove rows where event_type is minutes and date is before 2015-08-01
norm_inflation_sentiment = norm_inflation_sentiment[~((norm_inflation_sentiment['event_type'] == 'minutes') & (norm_inflation_sentiment['event_date'] < '2015-08-01'))].reset_index(drop=True)

################
# (ii) ILS data
################

# Normalise Date
ils_data["Date"] = ils_data["Date"].dt.normalize()

# Sort and set index
ils_data = ils_data.sort_values("Date").reset_index(drop=True)

# Convert to decimal and compute daily changes
ils_data["ils_1y"] = ils_data["Midpoint"] / 100.0
ils_data["ils_1y_chg"] = ils_data["ils_1y"].diff()
ils_data["ils_1y_chg_bps"] = ils_data["Midpoint"].diff() * 100

# Keep only Date and ils_1y_chg_bps
ils_data = ils_data[["Date", "ils_1y_chg_bps"]]

# Remove the 1st of each month to avoid month end jumps
ils_data = ils_data[~ils_data['Date'].dt.is_month_start].reset_index(drop=True)

# Remove any daily change greater than 3 standard deviations from the mean
threshold_bps = 3 * ils_data["ils_1y_chg_bps"].std()
print(f"Removing outliers beyond ±{threshold_bps:.1f} basis points")
ils_data_clean = ils_data[ils_data["ils_1y_chg_bps"].abs() <= threshold_bps].copy()
print(f"Removed {len(ils_data) - len(ils_data_clean)} outliers from {len(ils_data)} observations")

#################
# (iii) GLC data
##################

# Clean 1Y: strip spaces, remove thousands commas, coerce to float
glc["1Y"] = (
    glc["1Y"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)
glc["1Y"] = pd.to_numeric(glc["1Y"], errors="coerce")

# Normalise Date
glc["Date"] = pd.to_datetime(glc["Date"], errors="coerce")

# Sort and build decimal + change
glc = glc.sort_values("Date").copy()
glc["y_1y"] = glc["1Y"] / 100.0      # now works
glc["y_1y_chg"] = glc["y_1y"].diff()

# daily change in bps
glc["gilt_y_1y_chg_bps"] = glc["1Y"].astype(float).diff() * 100

##########################
# (iv) CPI Surprise data
##########################

# Coerce Date_cpi to datetime and make midnight
cpi_surprise["Date_cpi"] = pd.to_datetime(cpi_surprise["Date_cpi"], errors="coerce")
cpi_surprise["Date_cpi"] = cpi_surprise["Date_cpi"].dt.normalize()

# Create Daily CPI Surprise DataFrame
daily = ils_data.copy().reset_index(drop=True)

# Add CPI surprise column, default zero
daily["cpi_surprise"] = 0.0

# Add CPI release dummy
daily["cpi_release"] = 0

# Merge in monthly surprise values
for _, row in cpi_surprise.iterrows():
    d = row["Date_cpi"]   # release date
    s = row["surprise"]
    if d in daily["Date"].values:
        daily.loc[daily["Date"] == d, "cpi_surprise"] = s
        daily.loc[daily["Date"] == d, "cpi_release"] = 1

# %%
########### (3) Event Studies ###########

###########################
# (i) Baseline Events Study
###########################

# Merge all data into events_data

events_data = ils_data_clean.copy()

events_data = events_data.merge(norm_sentiment, left_on='Date', right_on='event_date', how='left')
events_data = events_data.merge(daily[['Date','cpi_surprise','cpi_release']], on='Date', how='left')
events_data = events_data.merge(glc[['Date','gilt_y_1y_chg_bps']], on='Date', how='left')
events_data = events_data.sort_values('Date').reset_index(drop=True)

# Keep only rows with an event
events_data = events_data.dropna(subset=['event_date']).reset_index(drop=True)

# %%
# Run Regression
df = events_data.copy()

# remove 2020-2022
# df = df[~df['Date'].between('2020-01-01', '2022-12-31')].reset_index(drop=True)

# Basic regression model
def run_baseline_regression(df, sentiment_var='norm_fomc_sent'):

    # Dependent variable
    y = df['ils_1y_chg_bps'] 
    
    # Independent variables
    X = pd.DataFrame({
        'mpc_member': df['is_mpc_member'],
        'event_type': pd.Categorical(df['event_type']).codes,
        'sentiment': df[sentiment_var],
        'cpi_surprise': df['cpi_surprise'], 
        'cpi_release': df['cpi_release'],
        'gilt_control': df['gilt_y_1y_chg_bps']
    })
    
    # Add constant
    X = sm.add_constant(X)
    
    # Remove rows with missing values
    regression_data = pd.concat([y, X], axis=1).dropna()
    y_clean = regression_data['ils_1y_chg_bps']
    X_clean = regression_data.drop('ils_1y_chg_bps', axis=1)
    
    # Run regression
    model = sm.OLS(y_clean, X_clean).fit()
    
    return model, len(regression_data)

# Run the baseline regression
model, n_obs = run_baseline_regression(df)

print("=== BASELINE EVENTS STUDY RESULTS ===")
print(f"Number of observations: {n_obs}")
print("\n" + model.summary().as_text())

# %%
# Export regression results to Markdown with raw LaTeX block
from stargazer.stargazer import Stargazer

# Run regression
model, n_obs = run_baseline_regression(df)

# Make Stargazer object
stargazer = Stargazer([model])

# Rename independent variables
pretty_names = {
    "const": "Constant",
    "mpc_member": "MPC member",
    "event_type": "Event type",
    "sentiment": "Sentiment",
    "cpi_surprise": "CPI surprise",
    "cpi_release": "CPI release",
    "gilt_control": "Gilt control"
}
stargazer.rename_covariates(pretty_names)

# Rename dependent variable
stargazer.dependent_variable_name("Change in 1Y Inflation Swap Rate (bps)")

# Add title only (no buggy custom notes)
stargazer.title("Baseline Events Study Regression")

# Get LaTeX table string
latex_table = stargazer.render_latex()

# Wrap in Pandoc raw LaTeX block
wrapped = f"```{{=latex}}\n{latex_table}\n```"

# Save as .md file
with open("Outputs/baseline_regression.md", "w") as f:
    f.write(wrapped)


# %%

###########################
# (ii) Growth Events Study
###########################

# Growth events data
growth_events = ils_data_clean.copy()

growth_events = growth_events.merge(norm_growth_sentiment, left_on='Date', right_on='event_date', how='left')
growth_events = growth_events.merge(daily[['Date','cpi_surprise','cpi_release']], on='Date', how='left')
growth_events = growth_events.merge(glc[['Date','gilt_y_1y_chg_bps']], on='Date', how='left')
growth_events = growth_events.sort_values('Date').reset_index(drop=True)

# Keep only rows with an event
growth_events = growth_events.dropna(subset=['event_date']).reset_index(drop=True)

# %%
# Run Regression
df = growth_events.copy()

# remove 2020-2022
# df = df[~df['Date'].between('2020-01-01', '2022-12-31')].reset_index(drop=True)

# Basic regression model
def run_baseline_regression(df, sentiment_var='norm_fomc_sent'):

    # Dependent variable
    y = df['ils_1y_chg_bps'] 
    
    # Independent variables
    X = pd.DataFrame({
        'mpc_member': df['is_mpc_member'],
        'event_type': pd.Categorical(df['event_type']).codes,
        'sentiment': df[sentiment_var],
        'cpi_surprise': df['cpi_surprise'], 
        'cpi_release': df['cpi_release'],
        'gilt_control': df['gilt_y_1y_chg_bps']
    })
    
    # Add constant
    X = sm.add_constant(X)
    
    # Remove rows with missing values
    regression_data = pd.concat([y, X], axis=1).dropna()
    y_clean = regression_data['ils_1y_chg_bps']
    X_clean = regression_data.drop('ils_1y_chg_bps', axis=1)
    
    # Run regression
    model = sm.OLS(y_clean, X_clean).fit()
    
    return model, len(regression_data)

# Run the baseline regression
model, n_obs = run_baseline_regression(df)

print("=== GROWTH EVENTS STUDY RESULTS ===")
print(f"Number of observations: {n_obs}")
print("\n" + model.summary().as_text())

# %%

# Run regression
model, n_obs = run_baseline_regression(df)

# Make Stargazer object
stargazer = Stargazer([model])

# Rename independent variables
pretty_names = {
    "const": "Constant",
    "mpc_member": "MPC member",
    "event_type": "Event type",
    "sentiment": "Sentiment",
    "cpi_surprise": "CPI surprise",
    "cpi_release": "CPI release",
    "gilt_control": "Gilt control"
}
stargazer.rename_covariates(pretty_names)

# Rename dependent variable
stargazer.dependent_variable_name("Change in 1Y Inflation Swap Rate (bps)")

# Add title only (no buggy custom notes)
stargazer.title("Growth Events Study Regression")

# Get LaTeX table string
latex_table = stargazer.render_latex()

# Wrap in Pandoc raw LaTeX block
wrapped = f"```{{=latex}}\n{latex_table}\n```"

# Save as .md file
with open("Outputs/growth_regression.md", "w") as f:
    f.write(wrapped)


# %%

##############################
# (iii) Inflation Events Study
##############################


# Inflation events data
inflation_events = ils_data_clean.copy()

inflation_events = inflation_events.merge(norm_inflation_sentiment, left_on='Date', right_on='event_date', how='left')
inflation_events = inflation_events.merge(daily[['Date','cpi_surprise','cpi_release']], on='Date', how='left')
inflation_events = inflation_events.merge(glc[['Date','gilt_y_1y_chg_bps']], on='Date', how='left')
inflation_events = inflation_events.sort_values('Date').reset_index(drop=True)
# Keep only rows with an event
inflation_events = inflation_events.dropna(subset=['event_date']).reset_index(drop=True)

# %%
# Run Regression
df = inflation_events.copy()

# remove 2020-2022
# df = df[~df['Date'].between('2020-01-01', '2022-12-31')].reset_index(drop=True)

# Basic regression model
def run_baseline_regression(df, sentiment_var='norm_fomc_sent'):

    # Dependent variable
    y = df['ils_1y_chg_bps'] 
    
    # Independent variables
    X = pd.DataFrame({
        'mpc_member': df['is_mpc_member'],
        'event_type': pd.Categorical(df['event_type']).codes,
        'sentiment': df[sentiment_var],
        'cpi_surprise': df['cpi_surprise'], 
        'cpi_release': df['cpi_release'],
        'gilt_control': df['gilt_y_1y_chg_bps']
    })
    
    # Add constant
    X = sm.add_constant(X)
    
    # Remove rows with missing values
    regression_data = pd.concat([y, X], axis=1).dropna()
    y_clean = regression_data['ils_1y_chg_bps']
    X_clean = regression_data.drop('ils_1y_chg_bps', axis=1)
    
    # Run regression
    model = sm.OLS(y_clean, X_clean).fit()
    
    return model, len(regression_data)

# Run the baseline regression
model, n_obs = run_baseline_regression(df)

print("=== INFLATION EVENTS STUDY RESULTS ===")
print(f"Number of observations: {n_obs}")
print("\n" + model.summary().as_text())

# %%

# Run regression
model, n_obs = run_baseline_regression(df)

# Make Stargazer object
stargazer = Stargazer([model])

# Rename independent variables
pretty_names = {
    "const": "Constant",
    "mpc_member": "MPC member",
    "event_type": "Event type",
    "sentiment": "Sentiment",
    "cpi_surprise": "CPI surprise",
    "cpi_release": "CPI release",
    "gilt_control": "Gilt control"
}
stargazer.rename_covariates(pretty_names)

# Rename dependent variable
stargazer.dependent_variable_name("Change in 1Y Inflation Swap Rate (bps)")

# Add title only (no buggy custom notes)
stargazer.title("Inflation Events Study Regression")

# Get LaTeX table string
latex_table = stargazer.render_latex()

# Wrap in Pandoc raw LaTeX block
wrapped = f"```{{=latex}}\n{latex_table}\n```"

# Save as .md file
with open("Outputs/inflation_regression.md", "w") as f:
    f.write(wrapped)

# %%
##############################
# (iv) Comparison Tables
##############################

# Summary Comparison Table Across All Event Types

def create_sentiment_comparison_table():

    # Define datasets and their names
    datasets = {
        'Baseline Events': events_data,
        'Inflation Events': inflation_events, 
        'Growth Events': growth_events
    }
    
    # Define sentiment measures with clean names
    sentiment_measures = {
        'Loughran-McDonald': 'norm_lm_sent',
        'ECB Lexicon': 'norm_ecb_sent', 
        'FinBERT': 'norm_finbert_sent',
        'FOMC-RoBERTa': 'norm_fomc_sent'
    }
    
    # Initialize results storage
    results = []
    
    # Loop through each dataset and sentiment measure
    for dataset_name, df in datasets.items():
        for measure_name, measure_var in sentiment_measures.items():
            try:
                model, n_obs = run_baseline_regression(df, measure_var)
                
                # Extract key statistics
                coef = model.params['sentiment']
                pval = model.pvalues['sentiment']
                r_squared = model.rsquared
                significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
                
                results.append({
                    'Event Type': dataset_name,
                    'Sentiment Measure': measure_name,
                    'Coefficient': f"{coef:.3f}{significance}",
                    'P-value': f"{pval:.3f}",
                    'R-squared': f"{r_squared:.3f}",
                    'N': n_obs
                })
                
            except Exception as e:
                # Handle any errors
                results.append({
                    'Event Type': dataset_name,
                    'Sentiment Measure': measure_name,
                    'Coefficient': 'Error',
                    'P-value': 'Error',
                    'R-squared': 'Error', 
                    'N': 'Error'
                })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(results)
    
    return comparison_df

# Create the comparison table
comparison_table = create_sentiment_comparison_table()

print("=== SENTIMENT MEASURE COMPARISON ACROSS EVENT TYPES ===")
print(comparison_table.to_string(index=False))

# %%
pivot_table = comparison_table.pivot(index='Sentiment Measure', 
                                   columns='Event Type', 
                                   values='Coefficient')

print("\n=== PIVOT TABLE: COEFFICIENTS BY SENTIMENT MEASURE ===")
print(pivot_table.to_string())

# %%
# Export comparison and pivot tables to Excel
with pd.ExcelWriter('Outputs/sentiment_regression_comparison.xlsx') as writer:
    comparison_table.to_excel(writer, sheet_name='Comparison Table', index=False)
    pivot_table.to_excel(writer, sheet_name='Pivot Table')

# %%
########### (4) Descriptive Statistics ###########

##########################
### (i) ILS Changes
##########################

# Summary statistics for ILS changes
print("\n=== ILS Change Summary Statistics ===")
print(ils_data_clean['ils_1y_chg_bps'].describe())

# Histogram of ILS changes
plt.figure(figsize=(10,6))
plt.hist(ils_data_clean['ils_1y_chg_bps'], bins=50, color='skyblue', edgecolor='black')
plt.title('Histogram of Daily 1Y ILS Changes (bps)')
plt.xlabel('Daily Change (bps)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

# %%
##########################
### (ii) Sentiment Scores
##########################

# Define clean column and index names
clean_names = {
    'norm_lm_sent': 'Loughran-McDonald',
    'norm_ecb_sent': 'ECB Lexicon', 
    'norm_finbert_sent': 'FinBERT',
    'norm_fomc_sent': 'FOMC-RoBERTa'
}

# Calculate correlation matrices
corr_matrix_events = events_data[['norm_lm_sent', 'norm_ecb_sent', 'norm_finbert_sent', 'norm_fomc_sent']].corr()
corr_matrix_growth = growth_events[['norm_lm_sent', 'norm_ecb_sent', 'norm_finbert_sent', 'norm_fomc_sent']].corr()
corr_matrix_inflation = inflation_events[['norm_lm_sent', 'norm_ecb_sent', 'norm_finbert_sent', 'norm_fomc_sent']].corr()

# Clean up names and round to 3 decimal places
corr_matrix_events_clean = corr_matrix_events.rename(columns=clean_names, index=clean_names).round(3)
corr_matrix_growth_clean = corr_matrix_growth.rename(columns=clean_names, index=clean_names).round(3)
corr_matrix_inflation_clean = corr_matrix_inflation.rename(columns=clean_names, index=clean_names).round(3)

# Export to Excel with multiple sheets
with pd.ExcelWriter('Outputs/sentiment_correlations.xlsx') as writer:
    corr_matrix_events_clean.to_excel(writer, sheet_name='Baseline Events')
    corr_matrix_growth_clean.to_excel(writer, sheet_name='Growth Events') 
    corr_matrix_inflation_clean.to_excel(writer, sheet_name='Inflation Events')



# %%

#################################
### (iii) Composition of Events
##################################

# Sample Composition Table with all three datasets
composition_table = pd.DataFrame({
    'Category': [
        'Total Events',
        'Speeches', 
        'Minutes',
        'MPC Member Communications',
        'Non-MPC Communications',
        'Sample Period'
    ],
    'Baseline Events': [
        f"{len(events_data)}",
        f"{len(events_data[events_data['event_type'] == 'speech'])} ({len(events_data[events_data['event_type'] == 'speech'])/len(events_data)*100:.1f}%)",
        f"{len(events_data[events_data['event_type'] == 'minutes'])} ({len(events_data[events_data['event_type'] == 'minutes'])/len(events_data)*100:.1f}%)",
        f"{len(events_data[events_data['is_mpc_member'] == 1])} ({len(events_data[events_data['is_mpc_member'] == 1])/len(events_data)*100:.1f}%)",
        f"{len(events_data[events_data['is_mpc_member'] == 0])} ({len(events_data[events_data['is_mpc_member'] == 0])/len(events_data)*100:.1f}%)",
        f"{events_data['event_date'].min().strftime('%B %Y')} to {events_data['event_date'].max().strftime('%B %Y')}"
    ],
    'Inflation Events': [
        f"{len(inflation_events)}",
        f"{len(inflation_events[inflation_events['event_type'] == 'speech'])} ({len(inflation_events[inflation_events['event_type'] == 'speech'])/len(inflation_events)*100:.1f}%)",
        f"{len(inflation_events[inflation_events['event_type'] == 'minutes'])} ({len(inflation_events[inflation_events['event_type'] == 'minutes'])/len(inflation_events)*100:.1f}%)",
        f"{len(inflation_events[inflation_events['is_mpc_member'] == 1])} ({len(inflation_events[inflation_events['is_mpc_member'] == 1])/len(inflation_events)*100:.1f}%)",
        f"{len(inflation_events[inflation_events['is_mpc_member'] == 0])} ({len(inflation_events[inflation_events['is_mpc_member'] == 0])/len(inflation_events)*100:.1f}%)",
        f"{inflation_events['event_date'].min().strftime('%B %Y')} to {inflation_events['event_date'].max().strftime('%B %Y')}"
    ],
    'Growth Events': [
        f"{len(growth_events)}",
        f"{len(growth_events[growth_events['event_type'] == 'speech'])} ({len(growth_events[growth_events['event_type'] == 'speech'])/len(growth_events)*100:.1f}%)",
        f"{len(growth_events[growth_events['event_type'] == 'minutes'])} ({len(growth_events[growth_events['event_type'] == 'minutes'])/len(growth_events)*100:.1f}%)",
        f"{len(growth_events[growth_events['is_mpc_member'] == 1])} ({len(growth_events[growth_events['is_mpc_member'] == 1])/len(growth_events)*100:.1f}%)",
        f"{len(growth_events[growth_events['is_mpc_member'] == 0])} ({len(growth_events[growth_events['is_mpc_member'] == 0])/len(growth_events)*100:.1f}%)",
        f"{growth_events['event_date'].min().strftime('%B %Y')} to {growth_events['event_date'].max().strftime('%B %Y')}"
    ]
})

print("Table X: Sample Composition")
print("=" * 80)
print(composition_table.to_string(index=False))

# Export to Excel
composition_table.to_excel('Outputs/sample_composition_table.xlsx', index=False)

# %%











# %%


# %%
import matplotlib.pyplot as plt

Date = ils_data["Date"]
ils_1y_chg_bps = ils_data["ils_1y_chg_bps"]

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(Date, ils_1y_chg_bps, label='1Y ILS Change (bps)', color='blue')

# Define event windows
events = [
    ("2016-06-01", "2017-06-01", "Brexit", "orange"),
    ("2020-03-01", "2021-12-31", "COVID-19", "yellow"),
    ("2022-09-01", "2022-11-01", "Truss Crisis", "purple")
]

# Add spans + centered labels
for start, end, label, color in events:
    ax.axvspan(start, end, alpha=0.3, color=color)
    midpoint = pd.to_datetime(start) + (pd.to_datetime(end) - pd.to_datetime(start)) / 2
    ax.text(midpoint, 175, label, fontsize=10, color="black", ha="center", va="top")

# Labels and title
ax.set_xlabel("Date")
ax.set_ylabel("Daily Change in 1Y ILS (bps)")
ax.set_title("Daily Changes in 1Y Inflation-Linked Swap Rates with Key Events")

# Fix y-axis between +150 and -250
ax.set_ylim(-250, 200)

plt.tight_layout()
plt.show()


# %%



# (4) Plot ILS Changes Over Time with Crisis Periods Highlighted
import matplotlib.pyplot as plt

Date = ils_data["Date"]
ils_1y_chg_bps = ils_data["ils_1y_chg_bps"]

# Plot ILS changes over time with crisis periods shaded
plt.figure(figsize=(12,6))
plt.plot(Date, ils_1y_chg_bps, label='1Y ILS Change (bps)', color='blue')
plt.axvspan('2016-06-01', '2017-06-01', alpha=0.3, color='orange', label='Brexit Period')
plt.axvspan('2020-03-01', '2021-12-31', alpha=0.3, color='yellow', label='COVID')
plt.axvspan('2022-09-01', '2022-11-01', alpha=0.3, color='purple', label='Truss Crisis')


# %%
Date = ils_data_clean["Date"]
ils_1y_chg_bps = ils_data_clean["ils_1y_chg_bps"]

# Plot ILS changes over time with crisis periods shaded
plt.figure(figsize=(12,6))
plt.plot(Date, ils_1y_chg_bps, label='1Y ILS Change (bps)', color='blue')
plt.axvspan('2016-06-01', '2017-06-01', alpha=0.3, color='orange', label='Brexit Period')
plt.axvspan('2020-03-01', '2021-12-31', alpha=0.3, color='yellow', label='COVID')
plt.axvspan('2022-09-01', '2022-11-01', alpha=0.3, color='purple', label='Truss Crisis')
# %%