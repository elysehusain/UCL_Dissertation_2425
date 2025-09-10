# Metadata for Bank of England MPC meeting minutes

# %%

# Import libraries
import pandas as pd
import os
import re
from datetime import datetime

# %% 

###### (1) MINUTES METADATA ######

# Read minutes metadata from CSV
minutes_metadata = pd.read_csv('All_data/Metadata/minutes_metadata.csv')

# Keep only pdf's
minutes_metadata = minutes_metadata[minutes_metadata['url'].str.endswith('.pdf')]

# Path to directory containing the minutes text files
minutes_path = "All_data/BoE_Minutes"

# Checks 
# Count number of .pdf files in the input directory
pdf_count = len([f for f in os.listdir(minutes_path) if f.endswith('.txt')])
print(f"Number of PDF files in '{minutes_path}': {pdf_count}")

# Count number of empty txt files in the input directory
empty_txt_count = len([f for f in os.listdir(minutes_path) if f.endswith('.txt') and os.path.getsize(os.path.join(minutes_path, f)) == 0])
print(f"Number of empty TXT files in '{minutes_path}': {empty_txt_count}")

# %% 

###### (2) EXTRACT PUBLICATION DATE FROM MINUTES ######

def extract_publication_date(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        # Method 1: Look for "Publication date: " in first 10 lines
        for line in lines[:10]:
            if "Publication date: " in line:
                date_part = line.split("Publication date: ")[1].strip()
                return date_part
        
        # Method 2: Find second date in first 10 lines
        date_pattern = r'\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        dates_found = []
        
        for line in lines[:10]:
            matches = re.findall(date_pattern, line, re.IGNORECASE)
            dates_found.extend(matches)
        
        # Return second date if found
        if len(dates_found) >= 2:
            return dates_found[1]
        elif len(dates_found) == 1:
            return dates_found[0]
        
        return None
        
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

# Create publication_date column
publication_dates = []

for index, row in minutes_metadata.iterrows():
    filename = row['filename']
    
    # Convert .pdf filename to .txt
    if filename.endswith('.pdf'):
        txt_filename = filename.replace('.pdf', '.txt')
        file_path = os.path.join(minutes_path, txt_filename)
        
        if os.path.exists(file_path):
            pub_date = extract_publication_date(file_path)
            publication_dates.append(pub_date)
        else:
            publication_dates.append(None)
    else:
        publication_dates.append(None)

# Add the new column
minutes_metadata['publication_date'] = publication_dates

print(f"\nExtracted publication dates for {len([d for d in publication_dates if d is not None])} files")

# Convert to datetime format
def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%d %B %Y')
    except ValueError:
        return None

minutes_metadata['publication_date'] = minutes_metadata['publication_date'].apply(parse_date)

# %%

###### (3) CLEAN MINUTES METADATA ######

# 1 Duplicate date found - transcript

# Remove row with transcript in filename
minutes_metadata = minutes_metadata[~minutes_metadata['filename'].str.contains('transcript', case=False)]

# %%

###### (4) CONVERT .TXT FILES TO CSV ######

# Read all text files 
import glob
text_files = glob.glob(os.path.join(minutes_path, '*.txt'))
data = []

# For loop
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        filename = os.path.basename(file)
        data.append({'filename': filename, 'content': content})
text_df = pd.DataFrame(data)

# Remove '.txt' from filenames
text_df['filename'] = text_df['filename'].str.replace('.txt', '', regex=False)

# Remove '.pdf' from filenames in minutes_metadata
minutes_metadata['filename'] = minutes_metadata['filename'].str.replace('.pdf', '', regex=False)

# Retain only entries that appear in minutes_metadata
text_df = text_df[text_df['filename'].isin(minutes_metadata['filename'])]


# %% 

###### (5) EXTRACT SUMMARY AND MINUTES FROM TEXT ######

# (i) Extract summary + minutes 
def extract_summary_minutes(text):
    start_match = re.search(r'Monetary Policy Summary, ', text, re.IGNORECASE)
    if start_match:
        start = start_match.start()
        return text[start:].strip()
    return None

text_df['summary_minutes'] = text_df['content'].apply(extract_summary_minutes)

# %%

# (ii) Extract summary_alone
def extract_summary(text):
    if pd.isna(text):
        return None
    
    end_match = re.search(r'minutes of the (special )?monetary policy committee', text, re.IGNORECASE)
    if end_match:
        end = end_match.start()
        return text[:end].strip()
    return None

# Apply the function
text_df['summary_alone'] = text_df['summary_minutes'].apply(extract_summary)


# %% 

# (iii) Extract minutes_alone
def extract_minutes(row):
    summary_minutes = row['summary_minutes']
    content = row['content']
    
    # If summary_minutes is not NA, use the original logic
    if pd.notna(summary_minutes):
        start_match = re.search(r'minutes of the (special )?monetary policy committee', summary_minutes, re.IGNORECASE)
        if start_match:
            start = start_match.start()
            return summary_minutes[start:].strip()
        return None
    
    # If summary_minutes is NA, find correct instance
    else:
        # Find all matches in content
        matches = list(re.finditer(r'minutes of the (special )?monetary policy committee', content, re.IGNORECASE))
        
        # If there are at least 3 matches, use the third one
        if len(matches) >= 3:
            start = matches[2].start()
            return content[start:].strip()
        # If there are 2 matches, use the second one
        elif len(matches) == 2:
            start = matches[1].start()
            return content[start:].strip()
        # If there's only 1 match, use it
        elif len(matches) == 1:
            start = matches[0].start()
            return content[start:].strip()
        
        return None

text_df['minutes_alone'] = text_df.apply(extract_minutes, axis=1)

# %%
###### (6) RENAME COLUMNS AND HANDLE MISSING SUMMARIES ######

# Change summary_minutes column name to 'full_text'
text_df = text_df.rename(columns={'summary_minutes': 'full_text'})

# Change summary_alone to text_summary
text_df = text_df.rename(columns={'summary_alone': 'text_summary'})

# Change minutes_alone to text_minutes
text_df = text_df.rename(columns={'minutes_alone': 'text_minutes'})

# If full_text is NA, but text_minutes is not NA, then set full_text to text_minutes
text_df['full_text'] = text_df.apply(lambda row: row['text_minutes'] if pd.isna(row['full_text']) and pd.notna(row['text_minutes']) else row['full_text'], axis=1)


# %%

###### (7) SAVE CLEANED DATA ######

# Minutes
minutes_metadata.to_csv('Final_data/cleaned_minutes_metadata.csv', index=False)

# Text
text_df.to_csv('Final_data/cleaned_minutes_texts.csv', index=False)

