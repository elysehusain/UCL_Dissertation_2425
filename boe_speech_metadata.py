# Metadata for Bank of England speeches data
# %%

# Import libraries
import os
import re
import pandas as pd

# %% 
####### (1) MPC MEMBERS CSV #######

# Read csv
mpc_members = pd.read_csv('All_data/MPC_members.csv')

# %%
# (i) Clean MPC members data

# Remove trailing whitespace from all columns
mpc_members = mpc_members.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Create surname column 
mpc_members['Surname'] = mpc_members['Name'].str.split().str[-1]

# Create first name column 
mpc_members['First_Name'] = mpc_members['Name'].str.split().str[:-1].str.join(' ')

# Remove any thing after first word of First_Name
mpc_members['First_Name'] = mpc_members['First_Name'].str.split().str[0]

# Create 'Other_Name' column for any other names
mpc_members['Other_Name'] = mpc_members['Name'].str.split().str[1:-1].str.join(' ')

# Remove any .'s or ( or ) from Other_Name
mpc_members['Other_Name'] = mpc_members['Other_Name'].str.replace(r'[().]', '', regex=True).str.strip()

# For 'Andrew G Haldane', put 'Andy' in Other_Name
mpc_members.loc[mpc_members['Name'] == 'Andrew G Haldane', 'Other_Name'] = 'Andy'
mpc_members.loc[mpc_members['Name'] == 'Catherine L. Mann', 'Other_Name'] = ''

# Rearrange columns
mpc_members = mpc_members[['Name', 'First_Name', 'Other_Name', 'Surname', 'Title', 'From', 'To']]

# %%
# (ii) Make sure 'From' and 'To' columns are in the correct format

# Make 'To' and 'From' columns datetime
mpc_members['From'] = pd.to_datetime(mpc_members['From'], errors='coerce')
mpc_members['To'] = pd.to_datetime(mpc_members['To'], errors='coerce')

# %%
# (iii) Save cleaned MPC members data to csv
mpc_members.to_csv('All_data/cleaned_mpc_members.csv', index=False)

# %% 
####### (2) SPEECH METADATA CSV #######

# Open speech metadata file
speech_metadata = pd.read_csv('All_data/Metadata/speech_metadata.csv')

# %%
# (i) Clean speech metadata

# Remove all appendices, annexes, slides
speech_metadata = speech_metadata[~speech_metadata['filename'].str.contains('appendix|annex|slides|chart|references', case=False, na=False)]
speech_metadata = speech_metadata[~speech_metadata['link_text'].str.contains('Q&A|slides|chart|appendix|references', case=False, na=False)]

# %%
# (ii) Extract speaker information

# Add new column for speaker, after "remarks by" or "speech by" in link_text
def extract_speaker(link_text):
    match = re.search(r'(?:remarks by|speech by)\s+(.+)', link_text, re.IGNORECASE)
    return match.group(1).strip() if match else None
speech_metadata['speaker'] = speech_metadata['link_text'].apply(extract_speaker)

# If speaker empty, use words after "by" in link_text
def extract_speaker_fallback(link_text):
    match = re.search(r'by\s+(.+)', link_text, re.IGNORECASE)
    return match.group(1).strip() if match else None
speech_metadata['speaker'] = speech_metadata['speaker'].fillna(speech_metadata['link_text'].apply(extract_speaker_fallback))

# Remove any leading or trailing whitespace from speaker names
speech_metadata['speaker'] = speech_metadata['speaker'].str.strip()

# Left with 503 speeches after initial cleaning

# %% 
# (iii) Clean speaker names

# Remove words "Sir" and "Dame" and "." from speaker names
def clean_speaker_name(name):
    if pd.isna(name):
        return name
    name = re.sub(r'\b(Sir|Dame)\b', '', name, flags=re.IGNORECASE).strip()
    name = re.sub(r'\.', '', name).strip()  # Remove any trailing periods
    return name

# Apply function
speech_metadata['speaker_clean'] = speech_metadata['speaker'].apply(clean_speaker_name)

# Remove initials from speaker names
def remove_initials(name):
    if pd.isna(name):
        return name
    
    # Convert to string and strip
    name = str(name).strip()
    
    # Remove standalone single letters
    name = re.sub(r'\s+[A-Z]\s+', ' ', name)
    
    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# Apply function
speech_metadata['speaker_clean'] = speech_metadata['speaker_clean'].apply(remove_initials)

# Get first and last name from speaker names
def get_first_last_name(name):
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    
    # Split by spaces
    words = name.split()
    
    if len(words) == 0:
        return None
    elif len(words) == 1:
        # Only one word - return as is
        return words[0]
    elif len(words) == 2:
        # Perfect - first and last name
        return ' '.join(words)
    else:
        # Hyphnated names or more than two words - take first and last
        return f"{words[0]} {words[1]}"

# Apply function
speech_metadata['speaker_clean'] = speech_metadata['speaker_clean'].apply(get_first_last_name)

# Print unique speakers after cleaning to check 
print("After taking first 2 words:")
print(speech_metadata['speaker_clean'].unique())

# %%
####### (3) SPEECH METADATA FROM .TXT FILES #######

# (i) Extract filename and date of speech from .txt files

# Define the path to the directory containing the text files
speeches_path = 'All_data/BoE_Speeches'

# Empty list to store metadata
speech_metadata_list = []

# Pattern to match dates like "12 November 2015", "12th November 2015", "3rd March 2020", "1st January 2021"
date_pattern = re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}\b")

for filename in os.listdir(speeches_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(speeches_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        text = "\n".join(lines)  # Combine all for full-text matching

        # Extract date
        date_match = date_pattern.search(text)
        date = date_match.group() if date_match else "Unknown"

        speech_metadata_list.append({
            "filename": filename,
            "date": date
        })

# Convert to df
speech_metadata_txt = pd.DataFrame(speech_metadata_list)

# %%
# (ii) Clean speech metadata from .txt files

# Remove unknown dates
speech_metadata_txt = speech_metadata_txt[speech_metadata_txt['date'].str.lower() != 'unknown']

# Remove 'th' 'rd' 'st' 'nd' from dates
speech_metadata_txt['date'] = speech_metadata_txt['date'].str.replace(r'\b(\d{1,2})(st|nd|rd|th)\b', r'\1', regex=True)

# Convert to datetime
speech_metadata_txt['date_cleaned'] = pd.to_datetime(speech_metadata_txt['date'], errors='coerce')

# %%
####### (4) MERGE DATASETS #######

# Remove .txt extension from filenames
speech_metadata_txt['filename'] = speech_metadata_txt['filename'].str.replace('.txt', '', regex=False)

# Remove .pdf extension from filenames in speech_metadata
speech_metadata['filename'] = speech_metadata['filename'].str.replace('.pdf', '', regex=False)

# Left join date from speech_metadata_txt to speech_metadata
speech_metadata_all = pd.merge(
    speech_metadata,
    speech_metadata_txt[['filename', 'date_cleaned']],
    on='filename',
    how='left',
    suffixes=('', '_txt')
)

# 503 speeches


# %%
####### (5) FINAL CLEANING AND SPEAKER EXTRACTION #######

# (i) Fill NA speakers with full names from speech_metadata_all

# Create list of all speakers
all_speakers = speech_metadata_all['speaker_clean'].unique()

# If speaker in speech_metadata_all is NaN, see if any full names from all_speakers appear in link_text, if not NaN return speaker
def find_speaker_in_link_text(row):
    speaker = str(row['speaker_clean']).strip()
    link_text = str(row['link_text']).strip()

    if pd.notna(speaker) and speaker != "None":
        return speaker

    for full_name in all_speakers:
        if pd.notna(full_name) and full_name.strip() in link_text:
            return full_name
    
    return pd.NA

# Apply the function to fill NaN speakers
speech_metadata_all['speaker_clean'] = speech_metadata_all.apply(
    find_speaker_in_link_text, axis=1
)

# Remove rows where speaker could not be identified
speech_metadata_all = speech_metadata_all[speech_metadata_all['speaker_MPC'].notna()]

# Print count of rows after cleaning
print("Count of rows after cleaning:", len(speech_metadata_all))

# 496 rows


# %%

# (ii) Identify MPC members in speech metadata

# Create First_Name and Surname columns for easier matching
speech_metadata_all['First_Name'] = speech_metadata_all['speaker_clean'].str.split().str[0]
speech_metadata_all['Surname'] = speech_metadata_all['speaker_clean'].str.split().str[1]

# If first_name and surname match with MPC members, return full name, else return "Not_MPC"
def match_speaker_to_mpc(row):
    if pd.isna(row['speaker_clean']):
        return pd.NA
    
    first_name = str(row['First_Name']).strip().lower()
    surname = str(row['Surname']).strip().lower()
    speech_date = pd.to_datetime(row['date_cleaned'])  # Get speech date from the row

    # Check if both first name and surname match any MPC member
    for _, mpc_row in mpc_members.iterrows():
        mpc_first_name = str(mpc_row['First_Name']).strip().lower()
        mpc_other_name = str(mpc_row['Other_Name']).strip().lower() if pd.notna(mpc_row['Other_Name']) else ""
        mpc_surname = str(mpc_row['Surname']).strip().lower()

        if (first_name == mpc_first_name or first_name == mpc_other_name) and surname == mpc_surname:
            # Check if speech date falls within tenure
            from_date = pd.to_datetime(mpc_row['From'])  # Get dates from mpc_row, not row
            to_date = pd.to_datetime(mpc_row['To']) if pd.notna(mpc_row['To']) else pd.Timestamp.now()
            
            if from_date <= speech_date <= to_date:
                return row['speaker_clean']  # Return full name
    
    return "Not_MPC"

# Apply the function to create 'speaker_MPC' column
speech_metadata_all['speaker_MPC'] = speech_metadata_all.apply(
    match_speaker_to_mpc, axis=1
)

# Create flag for MPC member 
speech_metadata_all['is_MPC_member'] = speech_metadata_all['speaker_MPC'].apply(
    lambda x: pd.NA if pd.isna(x) else (1 if x != "Not_MPC" else 0)
)

# Quick checks 
print("Not_MPC count:", (speech_metadata_all['speaker_MPC']=="Not_MPC").sum())
print("is_MPC_member=0 count:", (speech_metadata_all['is_MPC_member']==0).sum())
print("Speech count:" , len(speech_metadata_all))

# 503 rows


# %% 
# (iv) Add MPC titles to metadata

def get_title_for_date(first_name, surname, speech_date):
    if pd.isna(first_name) or pd.isna(surname):
        return None
    
    # Convert to strings and lowercase for matching
    first_name = str(first_name).strip().lower()
    surname = str(surname).strip().lower()
    
    # Filter for people with matching surname first
    surname_matches = mpc_members[mpc_members['Surname'].str.lower().str.strip() == surname]
    
    if surname_matches.empty:
        return None
    
    # From surname matches, check for first name or other name match
    for _, row in surname_matches.iterrows():
        mpc_first_name = str(row['First_Name']).strip().lower()
        mpc_other_name = str(row['Other_Name']).strip().lower() if pd.notna(row['Other_Name']) else ""
        
        # Check if first name matches either first name or other name
        if first_name == mpc_first_name or (mpc_other_name and first_name == mpc_other_name):
            # Check if speech date falls within tenure
            from_date = pd.to_datetime(row['From'])
            to_date = pd.to_datetime(row['To']) if pd.notna(row['To']) else pd.Timestamp.now()
            
            if from_date <= speech_date <= to_date:
                return row['Title']
    
    return None

# Apply the function using First_Name and Surname columns
speech_metadata_all['Title'] = speech_metadata_all.apply(
    lambda row: get_title_for_date(row['First_Name'], row['Surname'], row['date_cleaned']), 
    axis=1
)

# Count NAs in Title column
print("Count of NAs in Title column:", speech_metadata_all['Title'].isna().sum())

# Count number of non-MPC members
print("Count of non-MPC members:", (speech_metadata_all['is_MPC_member'] == 0).sum())

# %%
# (iv) Tidy data, remove uneccessary elements

# Keep only relevant columns
speech_metadata_all = speech_metadata_all[['filename', 'url', 'link_text', 'date_cleaned', 'speaker_MPC', 'is_MPC_member']]

# Remove rows with NA dates - all QA'd
speech_metadata_all = speech_metadata_all[speech_metadata_all['date_cleaned'].notna()]

# Count rows in final metadata
print("Final speech count:", len(speech_metadata_all))

# 493 rows as expected

# %%
##### (6) SAVE CLEANED METADATA #######

# Export to csv
speech_metadata_all.to_csv('Final_data/cleaned_speech_metadata.csv', index=False)


# %% 
####### (7) TEXT FILES TO CSV #######

# Read all text files in Speeches_path and save to a CSV

import glob
# Get all text files in the directory
text_files = glob.glob(os.path.join(speeches_path, '*.txt'))
# Create a list to hold the data
data = []
# Loop through each text file
for file in text_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        # Extract the filename without the path
        filename = os.path.basename(file)
        # Append to the data list
        data.append({'filename': filename, 'content': content})
# Convert to DataFrame
text_df = pd.DataFrame(data)


# %%
def clean_speech_content(content):
    # Split content into lines
    lines = content.split('\n')
    
    # Only search in the first 30 lines
    search_lines = 30
    
    # List of possible phrases to look for
    phrases = [
        "All speeches are available online at www.bankofengland.co.uk/speeches",
        "All speeches are available online at www.bankofengland.co.uk/speeches ",
        "All speeches are available on the Bank of England's website",
        "All speeches are available online at www.bankofengland.co.uk/publications/Pages/speeches/default.aspx",
        "Bank of England Page 2 Speech",
        "Bank of England Page 3 Speech",
        "Bank of England Page 2 Remarks",
        "Bank of England Page 3 Remarks",
        "@BoE_PressOffice",
        "/speeches",
        "Introduction",
    ]
    
    # Search through the first N lines
    for i, line in enumerate(lines[:search_lines]):
        # Check current line
        for phrase in phrases:
            if phrase in line:
                remaining_lines = lines[i+1:]
                return '\n'.join(remaining_lines).strip()
        
        # Check current line + next line combined (for multi-line phrases)
        if i < len(lines) - 1:
            combined_line = line + " " + lines[i+1]
            for phrase in phrases:
                if phrase in combined_line:
                    remaining_lines = lines[i+2:]  # Skip both lines
                    return '\n'.join(remaining_lines).strip()
    
    return content

# Apply the cleaning function
text_df['content_cleaned'] = text_df['content'].apply(clean_speech_content)
# %%
# If the first or second word is speech or introduction, remove the words
def remove_speech_or_intro(content):
    # Split the content into words
    words = content.split()
    
    # Check if the first or second word is "speech" or "introduction"
    if len(words) > 0 and words[0].lower() in ['speech', 'introduction', '1.']:
        return ' '.join(words[1:])
    elif len(words) > 1 and words[1].lower() in ['speech', 'introduction', '1.']:
        return ' '.join(words[:1] + words[2:])
    
    return content
# Apply the function to remove "speech" or "introduction" from the start
text_df['content_cleaned'] = text_df['content_cleaned'].apply(remove_speech_or_intro)

# %%

# Remove .txt extension from filenames
text_df['filename'] = text_df['filename'].str.replace('.txt', '', regex=False)

# Only keep filename's that appear in speech_metadata_all
text_df = text_df[text_df['filename'].isin(speech_metadata_all['filename'])]
# Reset index
text_df.reset_index(drop=True, inplace=True)

# %%
# Quick checks 

# Count the number of words in each content
text_df['word_count'] = text_df['content_cleaned'].apply(lambda x: len(x.split()))

# Print the first few rows in word_count order
print(text_df[['filename', 'word_count']].sort_values(by='word_count').head(10))

# Yep seems to have worked


# %%

# Count number of times 'Inflation' or 'inflation' appears in the content
def count_inflation_mentions(content):
    return content.lower().count('inflation')

# Apply the function to count inflation mentions
text_df['inflation_mentions'] = text_df['content_cleaned'].apply(count_inflation_mentions)

# Inflation as percentage of total mentions
def count_inflation_percentage(content):
    total_mentions = len(re.findall(r'\b\w+\b', content))  # Count total words
    if total_mentions == 0:
        return 0
    inflation_mentions = content.lower().count('inflation')
    return (inflation_mentions / total_mentions) * 100

# Apply the function to count inflation percentage
text_df['inflation_percentage'] = text_df['content_cleaned'].apply(count_inflation_percentage)

# Print the first few rows with inflation mentions
print(text_df[['filename', 'inflation_mentions', 'inflation_percentage']].sort_values(by='inflation_mentions', ascending=False).head(10))

# %%
# Save cleaned text data to CSV

text_df.to_csv('Final_data/cleaned_speech_texts.csv', index=False)

# %%
