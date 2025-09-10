# This script conducts text analysis on data from the Bank of England. 
# It aims to identify whether communication from the Bank of England impacts inflation expectations.

# %%
import pandas as pd
import numpy as np
from pathlib import Path
import pysentiment2 as ps
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from tqdm import tqdm

# %%
################# (1) LOAD ALL DATA #################

# Read minutes metadata
minutes_metadata = pd.read_csv('Final_data/cleaned_minutes_metadata.csv')

# Read minutes text data 
minutes_text = pd.read_csv('Final_data/cleaned_minutes_texts.csv')

# Read speech metadata
speech_metadata = pd.read_csv('Final_data/cleaned_speech_metadata.csv')

# Read speech text data
speech_text = pd.read_csv('Final_data/cleaned_speech_texts.csv')

# Read mpc bank rate decisions
bank_rate_decisions = pd.read_csv('Final_data/cleaned_bank_rate_decisions.csv')

# Load ECB dictionary
ecb = pd.read_csv("/Users/elysehusain/Elyse's Documents/Education/UCL/Dissertation/Final Project/Dictionaries/export_lexicon.csv", sep=";")

# %%
################# (2) PARSE DATES & FINAL CLEANING #################

# Parse dates more carefully
minutes_metadata['publication_date'] = pd.to_datetime(minutes_metadata['publication_date'], errors='coerce')

# For speeches, handle potential missing date_cleaned column
speech_metadata['date_cleaned'] = pd.to_datetime(speech_metadata['date_cleaned'], errors='coerce')

# Call text columns 'text_for_nlp' to combine later
minutes_text = minutes_text.rename(columns={'full_text': 'text_for_nlp'})
speech_text = speech_text.rename(columns={'content_cleaned': 'text_for_nlp'})

# Function to clean text more comprehensively
def clean_text_comprehensive(text):
    if not isinstance(text, str):
        return ""
    
    # Remove various types of whitespace and formatting issues
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ') 
    text = text.replace('\t', ' ')
    text = text.replace('\xa0', ' ')
    text = text.replace('\u2019', "'")
    text = text.replace('\u2018', "'")
    text = text.replace('\u201c', '"')
    text = text.replace('\u201d', '"')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2026', '...')
    
    # Remove extra whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

minutes_text['text_for_nlp'] = minutes_text['text_for_nlp'].apply(clean_text_comprehensive)
speech_text['text_for_nlp'] = speech_text['text_for_nlp'].apply(clean_text_comprehensive)


# %%
################# (3) BUILD A UNIFIED EVENTS TABLE #################

# Minutes: merge metadata and text
minutes_combined = minutes_metadata.merge(
    minutes_text[['filename', 'text_for_nlp']], 
    on='filename', 
    how='left'
)
minutes_combined = minutes_combined.rename(columns={'publication_date': 'event_date'})
minutes_combined['event_type'] = 'minutes'
minutes_combined['speaker'] = 'Committee'
minutes_combined['is_mpc_member'] = True

# Speeches: merge metadata and text
speeches_combined = speech_metadata.merge(
    speech_text[['filename', 'text_for_nlp']], 
    on='filename', 
    how='left'
)

# Rename columns to standardize
speeches_combined = speeches_combined.rename(columns={
    'date_cleaned': 'event_date', 
    'speaker_MPC': 'speaker',
    'is_MPC_member': 'is_mpc_member'
})
speeches_combined['event_type'] = 'speech'

# Combine into unified events table
events = pd.concat([
    minutes_combined[['event_date', 'event_type', 'filename', 'speaker', 'is_mpc_member', 'text_for_nlp']],
    speeches_combined[['event_date', 'event_type', 'filename', 'speaker', 'is_mpc_member', 'text_for_nlp']]
], ignore_index=True)

# Check as expected 
print(f"Total events loaded: {len(events)}")
print(f"Minutes: {len(events[events['event_type'] == 'minutes'])}")
print(f"Speeches: {len(events[events['event_type'] == 'speech'])}")


# %%
################# (4) CREATE LABELS FROM VOTING DATA #################

# Only keep first 3 columns of bank rate decisions
voting_long = bank_rate_decisions[['date', 'new_bank_rate', 'previous_bank_rate']].copy()

# Create column 'meeting_sentiment' based on whether new rate > previous rate = 'hawk', < = 'dove', = 'neutral'
def categorize_vote(row):
    if row['new_bank_rate'] > row['previous_bank_rate']:
        return 'increase'
    elif row['new_bank_rate'] < row['previous_bank_rate']:
        return 'reduce'
    else:
        return 'maintain'

voting_long['vote_category'] = voting_long.apply(categorize_vote, axis=1)
voting_long['meeting_date'] = pd.to_datetime(voting_long['date'], errors='coerce')
voting_long = voting_long.dropna(subset=['meeting_date']).sort_values('meeting_date').reset_index(drop=True)
voting_long = voting_long[['meeting_date', 'vote_category']]
print(f"Total voting records: {len(voting_long)}")

# Merge with events to get meeting-level sentiment
meeting_sentiment = voting_long.copy()
meeting_sentiment['meeting_stance'] = meeting_sentiment['vote_category'].map({
    'increase': 'hawk',
    'reduce': 'dove',
    'maintain': 'neutral'
})
print(f"Meeting sentiment records: {len(meeting_sentiment)}")


# %%
################# (5) ASSIGN LABELS TO EVENTS #################

# Merge events with meeting sentiment based on exact date - some will be NA
events_with_labels = events.merge(
    meeting_sentiment[['meeting_date', 'meeting_stance']], 
    left_on='event_date', 
    right_on='meeting_date', 
    how='left'
)

# %%
################# (6) LM DICTIONARY #################

# %%
### (i) Loughran-McDonald Dictionary Sentiment Scoring ###

# Load Loughran-McDonald dictionary via pysentiment2
lm = ps.LM()

# Define function to score sentiment using LM dictionary
def score_lm_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return pd.Series({'positive_count': 0, 'negative_count': 0, 'net_lm_sent': 0.0, 
                         'lm_polarity': 0.0, 'lm_subjectivity': 0.0})
    
    tokens = lm.tokenize(text)
    score = lm.get_score(tokens)
    
    positive_count = score['Positive']
    negative_count = score['Negative']
    
    # Calculate net sentiment (normalised by total sentiment words)
    total_sentiment_words = positive_count + negative_count
    if total_sentiment_words > 0:
        net_sentiment = (positive_count - negative_count) / total_sentiment_words
    else:
        net_sentiment = 0.0
    
    return pd.Series({
        'positive_count': positive_count,
        'negative_count': negative_count,
        'net_lm_sent': net_sentiment,
        'lm_polarity': score['Polarity'],
        'lm_subjectivity': score['Subjectivity']
    })

# %%
### (ii) LM Dictionary Application ###

# Apply function LM dictionary sentiment scoring
lm_sentiment_scores = events_with_labels['text_for_nlp'].apply(score_lm_sentiment)

# Add LM sentiment scores to events dataframe
events_with_lm = pd.concat([events_with_labels, lm_sentiment_scores], axis=1)

# %%
################# (7) ECB DICTIONARY #################

# %% 
### (i) Pre-process ECB dictionary ###

# Function to add n-gram groups to ECB dictionary
def add_ngram_groups(ecb_dict):

    ecb_copy = ecb_dict.copy()
    keyword_col = ecb_copy.columns[0]
    
    groups = []
    current_group = 'A'
    group_counter = 0
    
    previous_ngram = ""
    
    for idx, row in ecb_copy.iterrows():
        current_ngram = row[keyword_col]
        current_words = current_ngram.split()
        
        if idx == 0:
            # First row starts group A
            groups.append(current_group)
            previous_words = current_words
        else:
            previous_words = previous_ngram.split()
            
            # Check if current n-gram builds on the previous one
            if (len(current_words) > len(previous_words) and 
                current_words[:len(previous_words)] == previous_words):
                # Same group - builds on previous
                groups.append(current_group)
            else:
                # New group
                group_counter += 1
                current_group = chr(65 + group_counter % 26)  # A, B, C, etc.
                if group_counter >= 26:
                    current_group += str(group_counter // 26)  # AA, AB, etc. for more groups
                groups.append(current_group)
        
        previous_ngram = current_ngram
    
    ecb_copy['ngram_group'] = groups
    return ecb_copy

# Add groups to ECB dictionary
ecb_with_groups = add_ngram_groups(ecb)

# %%
### (ii) ECB Dictionary Function ###

# Create optimized lookups
ecb_keywords_set = set(ecb_with_groups[ecb.columns[0]].values)
ecb_lookup_dict = ecb_with_groups.set_index(ecb.columns[0]).to_dict('index')
ecb_group_lookup = dict(zip(ecb_with_groups[ecb.columns[0]], ecb_with_groups['ngram_group']))

# Define function to score sentiment using ECB dictionary
def score_ecb_sentiment(text, ecb_dict=ecb_with_groups):

    def preprocess_for_ecb_dict(text):
        tokens = word_tokenize(text.lower())
        tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if token.strip()]
        
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token]
        return stemmed_tokens
    
    tokens = preprocess_for_ecb_dict(text)
    
    # Find all possible matches
    all_matches = []
    for i in range(len(tokens)):
        for j in range(1, 11):
            if i+j <= len(tokens):
                ngram = ' '.join(tokens[i:i+j])
                if ngram in ecb_keywords_set:
                    group = ecb_group_lookup[ngram]
                    all_matches.append({
                        'ngram': ngram,
                        'start_pos': i,
                        'length': j,
                        'group': group
                    })
    
    # For each group, keep only the longest match
    group_matches = {}
    for match in all_matches:
        group = match['group']
        if group not in group_matches or match['length'] > group_matches[group]['length']:
            group_matches[group] = match
    
    # Score the final matches
    final_matches = list(group_matches.values())
    mp_acco = mp_neut = mp_rest = ec_nega = ec_neut = ec_posi = 0
    
    for match in final_matches:
        ngram = match['ngram']
        row = ecb_lookup_dict[ngram]
        mp_acco += row.get('mp_acco', 0)
        mp_neut += row.get('mp_neut', 0)
        mp_rest += row.get('mp_rest', 0)
        ec_nega += row.get('ec_nega', 0)
        ec_neut += row.get('ec_neut', 0)
        ec_posi += row.get('ec_posi', 0)
    
    # Calculate sentiment scores  
    total_mp_sentiment = mp_acco + mp_rest
    if total_mp_sentiment > 0:
        net_ecb_sentiment = (mp_acco - mp_rest) / total_mp_sentiment
    else:
        net_ecb_sentiment = 0.0

    return pd.Series({
        'mp_accommodative': mp_acco,
        'mp_restrictive': mp_rest, 
        'mp_neutral': mp_neut,
        'net_ecb_sentiment': net_ecb_sentiment,
        'total_matches': len(final_matches),
        'total_mp_words': total_mp_sentiment
    })

# %%
### (iii) Apply ECB Dictionary ###

# Use tqdm for progress bar
tqdm.pandas()

# Apply function ECB dictionary sentiment scoring
ecb_sentiment_scores = events['text_for_nlp'].progress_apply(score_ecb_sentiment)

# Add ECB sentiment scores to events dataframe
events_with_lm_ecb = pd.concat([events_with_lm, ecb_sentiment_scores], axis=1)

# Save intermediate results with both LM and ECB sentiment
events_with_lm_ecb.to_csv('Int_data/processed_events_with_lm_ecb_sentiment.csv', index=False)

# %%
################# (8) TRANSFORMER-BASED SENTIMENT (FINBERT) #################

# %%
### (i) FinBERT Sentiment Analysis with Sentence Chunking ###

def get_finbert_sentiment_sentences(texts):

    # Load FinBERT model
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert = pipeline("sentiment-analysis", 
                      model=model, 
                      tokenizer=tokenizer,
                      return_all_scores=True,
                      device=-1)
    
    sentiment_scores = []
    
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            sentiment_scores.append(0.0)
            continue
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text.strip())
        sentence_sentiments = []
        
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                # Truncate sentence if too long
                sentence_text = sentence.strip()

                # Encode to tokens
                tokens = tokenizer.encode(sentence_text, add_special_tokens=False)
                    
                # If too long, truncate and decode back to text
                if len(tokens) > 510:  # Leave room for special tokens (CLS, SEP)
                    truncated_tokens = tokens[:510]
                    sentence_clean = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    sentence_clean = sentence_text
                
                result = finbert(sentence_clean)
                scores_dict = {r['label']: r['score'] for r in result[0]}
                pos_score = scores_dict.get('positive', 0)
                neg_score = scores_dict.get('negative', 0)
                net_sentiment = pos_score - neg_score
                sentence_sentiments.append(net_sentiment)
        
        # Average across sentences
        avg_sentiment = sum(sentence_sentiments) / len(sentence_sentiments) if sentence_sentiments else 0.0
        sentiment_scores.append(avg_sentiment)
    
    return sentiment_scores

# %%
### (ii) Apply FinBERT Sentiment Analysis ###

# Use function
finbert_scores = events['text_for_nlp'].progress_apply(lambda x: get_finbert_sentiment_sentences([x])[0])

# Add to events dataframe
events_with_lm_ecb['finbert_sentiment'] = finbert_scores

# Save intermediate results with FinBERT sentiment
events_with_lm_ecb.to_csv('Int_data/processed_events_with_lm_ecb_finbert_sentiment.csv', index=False)

# %%
################# (9) TRANSFORMER - FOMC ROBERTA #################

# %%
### (i) FOMC-RoBERTa Sentiment Analysis with Sentence Chunking and Token Truncation ###

# Define function
def get_fomc_roberta_sentiment(texts):
    
    # Load FOMC-RoBERTa
    tokenizer = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa", do_lower_case=True, do_basic_tokenize=True)
    model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/FOMC-RoBERTa", num_labels=3)
    config = AutoConfig.from_pretrained("gtfintechlab/FOMC-RoBERTa")
    
    # Create classifier pipeline
    fomc_classifier = pipeline('text-classification', 
                              model=model, 
                              tokenizer=tokenizer, 
                              config=config, 
                              device=-1, 
                              framework="pt")
    
    sentiment_scores = []
    
    for text in texts:
        if not isinstance(text, str) or not text.strip():
            sentiment_scores.append(0.0)
            continue
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text.strip())
        sentence_sentiments = []
        
        for sentence in sentences:
            sentence_text = sentence.strip()
            if len(sentence_text) > 10:  # Skip very short sentences
                
                # Token-based truncation
                tokens = tokenizer.encode(sentence_text, add_special_tokens=False)
                
                # If too long, truncate and decode back to text
                if len(tokens) > 510:  # Leave room for special tokens
                    truncated_tokens = tokens[:510]
                    sentence_clean = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                else:
                    sentence_clean = sentence_text
                
                # Get classification result
                result = fomc_classifier(sentence_clean)
                
                # Convert labels to numeric scores based on FOMC model:
                # LABEL_0: Dovish (negative monetary policy stance) = -1
                # LABEL_1: Hawkish (positive monetary policy stance) = +1  
                # LABEL_2: Neutral = 0
                label = result[0]['label']
                confidence = result[0]['score']
                
                if label == 'LABEL_0':
                    score = -confidence
                elif label == 'LABEL_1':
                    score = confidence
                else:
                    score = 0.0
                
                sentence_sentiments.append(score)
        
        # Average across sentences
        avg_sentiment = sum(sentence_sentiments) / len(sentence_sentiments) if sentence_sentiments else 0.0
        sentiment_scores.append(avg_sentiment)
    
    return sentiment_scores


# %%
### (ii) Apply Function ###

# Apply function to all events
fomc_roberta_scores = events['text_for_nlp'].progress_apply(lambda x: get_fomc_roberta_sentiment([x])[0])

# Add to events dataframe
events_with_lm_ecb['fomc_roberta_sentiment'] = fomc_roberta_scores

# Save final results with FOMC sentiment
events_with_lm_ecb.to_csv('Final_data/processed_events_with_lm_ecb_finbert_fomc_sentiment.csv', index=False)

# %%
################# (10) TOPICS - INFLATION & GROWTH #################

# %%
### (i) Define Keywords & Function ###

# Keywords
inflation_keywords = ['inflation', 'inflationary', 'disinflation', 'disinflationary', 'deflation', 'deflationary', 'CPI', 'RPI', 'CPIH', 'PCE', 'price level', 'price levels', 'price growth', 'price pressures', 'price dynamics', 'price pressure', 'price dynamic', 'wage growth', 'wage pressures', 'wage inflation', 'wage pressure', 'core inflation', 'headline inflation', 'underlying inflation', 'inflation expectations', 'inflation expectation', 'inflation target', 'inflation targeting', 'cost pressures', 'cost pressure', 'cost-push', 'demand-pull', 'monetary accommodation', 'monetary tightening', 'purchasing power', 'real wages', 'real wage', 'forward guidance', 'quantitative easing', 'QE', 'policy rate', 'policy rates', 'base rate', 'base rates', 'interest rate', 'interest rates', 'Bank Rate']
growth_keywords = ['GDP', 'growth', 'economic growth', 'output', 'economic output', 'expansion', 'expansionary', 'contraction', 'contractionary', 'recovery', 'recession', 'slowdown', 'upturn', 'downturn', 'employment', 'unemployment', 'labour market', 'labor market', 'job market', 'productivity', 'productive capacity', 'output gap', 'demand', 'aggregate demand', 'consumer demand', 'business investment', 'economic activity', 'economic momentum', 'economic conditions', 'capacity utilization', 'capacity utilisation', 'spare capacity', 'business confidence', 'consumer confidence', 'fiscal stimulus', 'fiscal consolidation', 'forward guidance', 'quantitative easing', 'QE', 'policy rate', 'policy rates', 'base rate', 'base rates', 'interest rate', 'interest rates', 'Bank Rate']

# Keywords table for Appendix
max_length = max(len(inflation_keywords), len(growth_keywords))
keywords_table = pd.DataFrame({
    'Inflation Keywords': inflation_keywords + [None] * (max_length - len(inflation_keywords)),
    'Growth Keywords': growth_keywords + [None] * (max_length - len(growth_keywords))
})

# Replace "None" with empty strings for better CSV formatting
keywords_table = keywords_table.fillna('')

# Save keywords table
keywords_table.to_csv('Final_data/inflation_growth_keywords.csv', index=False)

# Function to extract sentences with keywords
def extract_sentences_with_keywords(text, keywords):
    """Extract sentences containing any of the specified keywords."""
    import nltk
    from nltk.tokenize import sent_tokenize
    
    if not isinstance(text, str) or not text.strip():
        return ""
    
    sentences = sent_tokenize(text)
    keyword_set = set(keywords)
    matched_sentences = [sent for sent in sentences if any(kw in sent.lower() for kw in keyword_set)]
    
    return " ".join(matched_sentences)

# %%
### (ii) Apply Function & Score Sentiment for Inflation ###

# Create new df
inflation_events = events_with_lm_ecb.copy()

# Extract sentences containing inflation terms
inflation_sentences = extract_sentences_with_keywords(inflation_events['text_for_nlp'], inflation_keywords)

# Add to dataframe
inflation_events['inflation_sentences'] = inflation_events['text_for_nlp'].apply(
    lambda x: extract_sentences_with_keywords(x, inflation_keywords)
)

# Apply dictionary functions to inflation sentences 
inf_sent_lm = inflation_events['inflation_sentences'].apply(score_lm_sentiment)
inf_sent_ecb = inflation_events['inflation_sentences'].apply(score_ecb_sentiment)

# Apply finbert function to inflation sentences
inf_sent_finbert = inflation_events['inflation_sentences'].progress_apply(lambda x: get_finbert_sentiment_sentences([x])[0])

# Apply fomc roberta function to inflation sentences
inf_sent_fomc = inflation_events['inflation_sentences'].progress_apply(lambda x: get_fomc_roberta_sentiment([x])[0])

# Add as columns to inflation_events
inflation_events['inflation_sentiment_lm'] = inf_sent_lm['net_lm_sent']
inflation_events['inflation_sentiment_ecb'] = inf_sent_ecb['net_ecb_sentiment']
inflation_events['inflation_sentiment_finbert'] = inf_sent_finbert
inflation_events['inflation_sentiment_fomc'] = inf_sent_fomc

# Save results with inflation-specific sentiment
inflation_events.to_csv('Final_data/inflation_events_with_lm_ecb_finbert_fomc_sentiment.csv', index=False)

# %%
### (iii) Apply Function & Score Sentiment for Growth ###

# Create new df
growth_events = events_with_lm_ecb.copy()

# Extract sentences containing growth terms
growth_events['growth_sentences'] = growth_events['text_for_nlp'].apply(
    lambda x: extract_sentences_with_keywords(x, growth_keywords)
)

# Apply dictionary functions to growth sentences
growth_sent_lm = growth_events['growth_sentences'].apply(score_lm_sentiment)
growth_sent_ecb = growth_events['growth_sentences'].apply(score_ecb_sentiment)

# Apply finbert function to growth sentences
growth_sent_finbert = growth_events['growth_sentences'].progress_apply(lambda x: get_finbert_sentiment_sentences([x])[0])

# Apply fomc roberta function to growth sentences
growth_sent_fomc = growth_events['growth_sentences'].progress_apply(lambda x: get_fomc_roberta_sentiment([x])[0])

# Add as columns to growth_events
growth_events['growth_sentiment_lm'] = growth_sent_lm['net_lm_sent']
growth_events['growth_sentiment_ecb'] = growth_sent_ecb['net_ecb_sentiment']
growth_events['growth_sentiment_finbert'] = growth_sent_finbert
growth_events['growth_sentiment_fomc'] = growth_sent_fomc

# Save final results with growth-specific sentiment
growth_events.to_csv('Final_data/growth_events_with_lm_ecb_finbert_fomc_sentiment.csv', index=False)


# %%
################# (11) QA OF SENTIMENT SCORES #################

# Read in the final processed data 
all_events = pd.read_csv('Final_data/processed_events_with_lm_ecb_finbert_fomc_sentiment.csv')
growth_events = pd.read_csv('Final_data/growth_events_with_lm_ecb_finbert_fomc_sentiment.csv')
inflation_events = pd.read_csv('Final_data/inflation_events_with_lm_ecb_finbert_fomc_sentiment.csv')


# %%
###################################
### (i) LM Dictionary Validation 
###################################

# %%
# Word-level validation function
def extract_lm_words(text, n_examples=3):
    """Extract positive and negative words identified by LM dictionary"""
    if not isinstance(text, str) or not text.strip():
        return {'positive_words': [], 'negative_words': []}
    
    tokens = lm.tokenize(text)
    
    # Find positive and negative words in the tokenized text
    positive_words = []
    negative_words = []
    
    for word in tokens:
        if word.upper() in lm._posset:
            positive_words.append(word)
        elif word in lm._posset:
            positive_words.append(word)
        
        if word.upper() in lm._negset:
            negative_words.append(word)
        elif word in lm._negset:
            negative_words.append(word)
    
    return {
        'positive_words': positive_words[:n_examples],
        'negative_words': negative_words[:n_examples]
    }

# Create word-level validation table
sample_indices = np.random.choice(len(events_with_lm_ecb), 5, replace=False)
word_validation_data = []

for idx in sample_indices:
    text = events_with_lm_ecb.iloc[idx]['text_for_nlp']
    score = events_with_lm_ecb.iloc[idx]['net_lm_sent']
    words = extract_lm_words(text)
    
    word_validation_data.append({
        'Sample ID': idx,
        'LM Score': round(score, 3),
        'Positive Words': ', '.join(words['positive_words']),
        'Negative Words': ', '.join(words['negative_words'])
    })

lm_word_validation_df = pd.DataFrame(word_validation_data)

# %%
# Sample Text Inspection for LM Dictionary

# Get texts with extreme scores
high_pos = events_with_lm_ecb.nlargest(2, 'net_lm_sent')
high_neg = events_with_lm_ecb.nsmallest(2, 'net_lm_sent')
neutral = events_with_lm_ecb[abs(events_with_lm_ecb['net_lm_sent']) < 0.1].sample(1)

text_validation_data = []

for category, df in [("High Positive", high_pos), ("High Negative", high_neg), ("Neutral", neutral)]:
    for _, row in df.iterrows():
        text_validation_data.append({
            'Category': category,
            'LM Score': round(row['net_lm_sent'], 3),
            'Event Type': row['event_type'],
            'Date': row['event_date'],
            'Text Excerpt': row['text_for_nlp'][:300] + "..."
        })

lm_text_validation_df = pd.DataFrame(text_validation_data)

# %%
# %%
###################################
### (ii) ECB Dictionary Validation 
###################################

# %%
# Word-level validation function
def extract_ecb_words(text, n_examples=3):
    
    def preprocess_for_ecb_dict(text):
        
        tokens = word_tokenize(text.lower())
        tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens if token.strip()]
        
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(token) for token in tokens if token]
        return stemmed_tokens
    
    tokens = preprocess_for_ecb_dict(text)
    
    # Find all possible matches
    all_matches = []
    for i in range(len(tokens)):
        for j in range(1, 11):
            if i+j <= len(tokens):
                ngram = ' '.join(tokens[i:i+j])
                if ngram in ecb_keywords_set:
                    group = ecb_group_lookup[ngram]
                    all_matches.append({
                        'ngram': ngram,
                        'start_pos': i,
                        'length': j,
                        'group': group
                    })
    
    # For each group, keep only the longest match
    group_matches = {}
    for match in all_matches:
        group = match['group']
        if group not in group_matches or match['length'] > group_matches[group]['length']:
            group_matches[group] = match
    
    final_matches = list(group_matches.values())
    
    # Separate by sentiment type
    accommodative_ngrams = []
    restrictive_ngrams = []
    
    for match in final_matches:
        ngram = match['ngram']
        row = ecb_lookup_dict[ngram]
        
        if row.get('mp_acco', 0) > 0:
            accommodative_ngrams.append(ngram)
        if row.get('mp_rest', 0) > 0:
            restrictive_ngrams.append(ngram)
    
    return {
        'accommodative_ngrams': accommodative_ngrams[:n_examples],
        'restrictive_ngrams': restrictive_ngrams[:n_examples]
    }

# Create word-level validation table
sample_indices = np.random.choice(len(events_with_lm_ecb), 5, replace=False)
ecb_word_validation_data = []

for idx in sample_indices:
    text = events_with_lm_ecb.iloc[idx]['text_for_nlp']
    score = events_with_lm_ecb.iloc[idx]['net_ecb_sentiment']
    ngrams = extract_ecb_words(text)
    
    ecb_word_validation_data.append({
        'Sample ID': idx,
        'ECB Score': round(score, 3),
        'Accommodative N-grams': ', '.join(ngrams['accommodative_ngrams']),
        'Restrictive N-grams': ', '.join(ngrams['restrictive_ngrams'])
    })

ecb_word_validation_df = pd.DataFrame(ecb_word_validation_data) 

# %%
# Sample Text Inspection for ECB Dictionary

# Get texts with extreme scores
high_pos = events_with_lm_ecb.nlargest(2, 'net_ecb_sentiment')
high_neg = events_with_lm_ecb.nsmallest(2, 'net_ecb_sentiment')
neutral = events_with_lm_ecb.iloc[[(events_with_lm_ecb['net_ecb_sentiment'] - 0.5).abs().idxmin()]]

ecb_text_validation_data = []

for category, df in [("High Accommodative", high_pos), ("High Restrictive", high_neg), ("Neutral", neutral)]:
    for _, row in df.iterrows():
        ecb_text_validation_data.append({
            'Category': category,
            'ECB Score': round(row['net_ecb_sentiment'], 3),
            'Event Type': row['event_type'],
            'Date': row['event_date'],
            'Text Excerpt': row['text_for_nlp'][:300] + "..."
        })

ecb_text_validation_df = pd.DataFrame(ecb_text_validation_data)

# %%
###################################
### (iii) FinBERT Validation 
###################################

# Get texts with extreme scores
finbert_high_pos = events_with_lm_ecb.nlargest(2, 'finbert_sentiment')
finbert_high_neg = events_with_lm_ecb.nsmallest(2, 'finbert_sentiment')
finbert_neutral = events_with_lm_ecb[abs(events_with_lm_ecb['finbert_sentiment']) < 0.1].sample(1)

finbert_validation_data = []

for category, df in [("High Positive", finbert_high_pos), ("High Negative", finbert_high_neg), ("Neutral", finbert_neutral)]:
    for _, row in df.iterrows():
        finbert_validation_data.append({
            'Category': category,
            'FinBERT Score': round(row['finbert_sentiment'], 3),
            'Event Type': row['event_type'],
            'Date': row['event_date'],
            'Text Excerpt': row['text_for_nlp'][:300] + "..."
        })

finbert_validation_df = pd.DataFrame(finbert_validation_data)

# %%
###################################
### (iv) FOMC-RoBERTa Validation 
###################################

# Get texts with extreme scores
fomc_high_pos = events_with_lm_ecb.nlargest(2, 'fomc_roberta_sentiment')
fomc_high_neg = events_with_lm_ecb.nsmallest(2, 'fomc_roberta_sentiment')
fomc_neutral = events_with_lm_ecb[abs(events_with_lm_ecb['fomc_roberta_sentiment']) < 0.1].sample(1)

fomc_validation_data = []

for category, df in [("High Hawkish", fomc_high_pos), ("High Dovish", fomc_high_neg), ("Neutral", fomc_neutral)]:
    for _, row in df.iterrows():
        fomc_validation_data.append({
            'Category': category,
            'FOMC RoBERTa Score': round(row['fomc_roberta_sentiment'], 3),
            'Event Type': row['event_type'],
            'Date': row['event_date'],
            'Text Excerpt': row['text_for_nlp'][:300] + "..."
        })

fomc_validation_df = pd.DataFrame(fomc_validation_data)

# %%
###################################
### (v) Meeting Stance Validation 
###################################

# Normalise sentiment scores to 0-1 range for comparison
def normalize_series(series):
    min_val = series.min()
    max_val = series.max()
    if max_val - min_val == 0:
        return series.apply(lambda x: 0.0)  # Avoid division by zero
    return (series - min_val) / (max_val - min_val)

# All events
all_events['norm_lm_sent'] = normalize_series(all_events['net_lm_sent'])
all_events['norm_transformer_sent'] = normalize_series(all_events['finbert_sentiment'])
all_events['norm_ecb_sent'] = normalize_series(all_events['net_ecb_sentiment'])
all_events['norm_fomc_sent'] = normalize_series(all_events['fomc_roberta_sentiment'])


# Take only events with known meeting stance
meeting_stance_events = all_events[all_events['meeting_stance'].notna()]

# Group by meeting_stance and calculate mean sentiment scores
meeting_stance_grouped = meeting_stance_events.groupby('meeting_stance')[[
    'norm_lm_sent',
    'norm_transformer_sent',
    'norm_ecb_sent',
    'norm_fomc_sent'
]].mean().reset_index()

# Create validation table
meeting_stance_validation_data = []
for _, row in meeting_stance_grouped.iterrows():
    meeting_stance_validation_data.append({
        'Meeting Stance': row['meeting_stance'],
        'Number of Events': len(meeting_stance_events[meeting_stance_events['meeting_stance'] == row['meeting_stance']]),
        'Mean LM Score': round(row['norm_lm_sent'], 3),
        'Mean FinBERT Score': round(row['norm_transformer_sent'], 3),
        'Mean ECB Score': round(row['norm_ecb_sent'], 3),
        'Mean FOMC RoBERTa Score': round(row['norm_fomc_sent'], 3)
    })

meeting_stance_validation_df = pd.DataFrame(meeting_stance_validation_data)

# %%

# Export all validation tables to Excel
with pd.ExcelWriter('Final_data/sentiment_validation_tables.xlsx') as writer:
    lm_word_validation_df.to_excel(writer, sheet_name='LM_Word_Validation', index=False)
    lm_text_validation_df.to_excel(writer, sheet_name='LM_Text_Validation', index=False)
    ecb_word_validation_df.to_excel(writer, sheet_name='ECB_Word_Validation', index=False)
    ecb_text_validation_df.to_excel(writer, sheet_name='ECB_Text_Validation', index=False)
    finbert_validation_df.to_excel(writer, sheet_name='FinBERT_Validation', index=False)
    fomc_validation_df.to_excel(writer, sheet_name='FOMC_RoBERTa_Validation', index=False)
    meeting_stance_validation_df.to_excel(writer, sheet_name='Meeting_Stance_Validation', index=False)

# %%

## End of script


