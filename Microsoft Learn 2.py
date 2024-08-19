#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd

# Load the datasets
train_data = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\Train.csv")
test_data = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\Test.csv")
sample_submission = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\SampleSubmission.csv")


# In[5]:


# Display the first few rows of each dataset
train_head = train_data.head()
test_head = test_data.head()
sample_submission_head = sample_submission.head()


# In[6]:


# Check for null values and basic info
train_info = train_data.info()
test_info = test_data.info()
sample_submission_info = sample_submission.info()


# In[7]:


train_head, test_head, sample_submission_head, train_info, test_info, sample_submission_info


# Train.csv
# 
# Total Entries: 73,072
# 
# Columns: tweet_id, text, location
# 
# Observations:
# Only 16,448 entries have text data.
# 
# 43,460 entries have location data.
# 
# Missing values are present in both text and location columns.
# 
# Test.csv
# 
# Total Entries: 2,942
# 
# Columns: tweet_id, text
# 
# Observations:
# No missing values in text.
# This dataset doesn't include locations (as expected since it's for prediction).
# 
# 
# SampleSubmission.csv
# 
# Total Entries: 2,942
# 
# Columns: tweet_id, location
# Observations:
# The location column is entirely empty, which is expected since it's where predictions should go.

# # Next Steps:
# 
# Handle Missing Data: We need to address the missing text and location data in the Train.csv file.
# 
# Identify Abbreviations: Write a function to identify abbreviations in the dataset.
# 
# Preprocessing: Prepare the data for model training by cleaning and tokenizing the text, handling abbreviations, and vectorizing.
# 
# Model Training: Train a model using the preprocessed data.

# In[8]:


# drop rowa where text is Nan in train data since those can't be used
train_data_cleaned = train_data.dropna(subset=['text'])


# In[9]:


# Function to identify abbreviations in the text

def find_abbreviations(text_series):
    abbreviations = {}
    for text in text_series:
        words = text.split()
        for word in words:
            if word.isupper() and len(word) <=5:
                if word in abbreviations:
                    abbreviations[word] +=1
                else:
                    abbreviations[word] = 1
    return abbreviations


# In[10]:


# Identify abbreviations in the cleaned train data
abbreviations_found = find_abbreviations(train_data_cleaned['text'])


# In[11]:


# Sort abbreviations by frequency
sorted_abbreviations = dict(sorted(abbreviations_found.items(), key=lambda item: item[1], reverse=True))
sorted_abbreviations


# The abbreviations found in the dataset include a wide variety of terms, some of which are likely locations (e.g., NYC, LA, DC, FL, TX), while others are common abbreviations not related to locations (e.g., RT, FEMA, BBC).
# 
# # Key Abbreviations Related to Locations:
# 
# NYC: New York City
# 
# LA: Los Angeles
# 
# DC: Washington, D.C.
# 
# FL: Florida
# 
# TX: Texas
# 
# NC: North Carolina
# 
# CA: California
# 
# GA: Georgia

# # Next Steps:
# 
# **Create a Mapping:** Develop a mapping dictionary that can convert common abbreviations into their full forms.
# 
# **Text Preprocessing:** Apply this mapping during text preprocessing to ensure that abbreviations are expanded before feeding the text into the model.
# 
# **Modeling:** Use the cleaned and expanded text data to train the model.
# 
# I'll now create a function to map these abbreviations to their full forms and apply this mapping to the dataset

# In[12]:


# Create a mapping dictionary for common location abbreviations

abbreviation_mapping = {
    'NYC': 'New York City',
    'LA': 'Los Angeles',
    'DC': 'Washington D.C.',
    'SF': 'San Francisco',
    'TX': 'Texas',
    'FL': 'Florida',
    'GA': 'Georgia',
    'NC': 'North Carolina',
    'CA': 'California',
    'PR': 'Puerto Rico',
    'UK': 'United Kingdom',
    'NZ': 'New Zealand',
    'SC': 'South Carolina',
    'UK': 'United Kingdom',
    'UAE': 'United Arab Emirates',
    'USA': 'United States',
    'UK': 'United Kingdom',
    'US': 'United States',
    # Add more abbreviations as necessary
}


# In[13]:


# Function to expand abbreviations in text
def expand_abbreviations(text, mapping):
    words = text.split()
    expanded_words = [mapping.get(word, word) for word in words]
    return ' '.join(expanded_words)


# In[14]:


# # Apply the mapping to the train and test datasets
# train_data_cleaned['text_expanded'] = train_data_cleaned['text'].apply(lambda x: expand_abbreviations(x, abbreviation_mapping))
# test_data['text_expanded'] = test_data['text'].apply(lambda x: expand_abbreviations(x, abbreviation_mapping))

# Apply the mapping to the train and test datasets using .loc to avoid the warning
train_data_cleaned.loc[:, 'text_expanded'] = train_data_cleaned['text'].apply(lambda x: expand_abbreviations(x, abbreviation_mapping))
test_data.loc[:, 'text_expanded'] = test_data['text'].apply(lambda x: expand_abbreviations(x, abbreviation_mapping))


# In[15]:


# Display the first few rows after expansion
train_data_expanded_head = train_data_cleaned[['tweet_id', 'text', 'text_expanded']].head()
test_data_expanded_head = test_data[['tweet_id', 'text', 'text_expanded']].head()
train_data_expanded_head, test_data_expanded_head


# The text data in both the Train.csv and Test.csv files have been successfully expanded to replace common abbreviations with their full forms. Here are a few examples:
# 
# # Train Data (After Expansion):
# 
# **Original:** "Flash floods struck a Maryland city on Sunday, ..."
# 
# **Expanded:** "Flash floods struck a Maryland city on Sunday, ..."
# 
# # Test Data (After Expansion):
# 
# **Original:** "What is happening to the infrastructure in NYC?"
# 
# **Expanded:** "What is happening to the infrastructure in New York City?"

# # Next Steps:
# 
# **Further Preprocessing:** Continue cleaning the text, such as removing URLs, punctuation, and handling other text normalization tasks.
# 
# **Feature Engineering:** Convert the text into features that a machine learning model can process (e.g., TF-IDF vectorization).
# 
# **Modeling:** Train and evaluate the model using the preprocessed data.
# 
# **Evaluation:** Use the Word Error Rate (WER) metric to evaluate model performance.

# In[16]:


# Let's proceed with further text preprocessing and feature engineering.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Further Preprocessing
# removing URLs, special characters and lower case
import re

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove mentions and hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text


# In[17]:


# train_data_cleaned['text_cleaned'] = train_data_cleaned['text_expanded'].apply(clean_text)
# test_data['text_cleaned'] = test_data['text_expanded'].apply(clean_text)

train_data_cleaned.loc[:, 'text_cleaned'] = train_data_cleaned['text_expanded'].apply(clean_text)
test_data.loc[:, 'text_cleaned'] = test_data['text_expanded'].apply(clean_text)


# In[18]:


# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(train_data_cleaned['text_cleaned'])
X_test_tfidf = vectorizer.transform(test_data['text_cleaned'])


# In[19]:


# Split the training data for validation
X_train, X_validation, y_train, y_validation = train_test_split(X_train_tfidf, train_data_cleaned['location'].fillna(''), test_size=0.2, random_state=42)


# In[20]:


# Display the shape of transformed data
X_train_shape = X_train.shape
X_validation_shape = X_validation.shape
X_test_shape = X_test_tfidf.shape

X_train_shape, X_validation_shape, X_test_shape


# In[21]:


from sklearn.linear_model import LogisticRegression

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)
logistic_model.fit(X_train, y_train)


# In[22]:


# Predictions on validation set
y_pred_validation = logistic_model.predict(X_validation)


# In[23]:


from sklearn.metrics import classification_report, f1_score

# Classification report
classification_report_validation = classification_report(y_validation, y_pred_validation)
f1_score_validation = f1_score(y_validation, y_pred_validation, average='weighted')


# In[30]:


# from jiwer import wer

# # Calculate WER
# def evaluate_wer(y_true, y_pred):
#     total_wer = 0
#     for true, pred in zip(y_true, y_pred):
#         total_wer += wer(true, pred)
#     return total_wer / len(y_true)

from jiwer import wer

# Calculate WER while handling empty strings
def evaluate_wer(y_true, y_pred):
    total_wer = 0
    count = 0
    for true, pred in zip(y_true, y_pred):
        if true.strip() and pred.strip():  # Ensure neither is an empty string
            total_wer += wer(true, pred)
            count += 1
    return total_wer / count if count > 0 else float('inf')  # Return 'inf' if all pairs are invalid

# Predictions on the training set
y_pred_train = logistic_model.predict(X_train)

# Calculate WER for training and validation sets
train_wer = evaluate_wer(y_train, y_pred_train)
validation_wer = evaluate_wer(y_validation, y_pred_validation)

# Display the WER values
print(f"Word Error Rate (WER) on training set: {train_wer:.4f}")
print(f"Word Error Rate (WER) on validation set: {validation_wer:.4f}")


# In[31]:


validation_wer = evaluate_wer(y_validation, y_pred_validation)


# In[32]:


# Predict on the test data
test_predictions = logistic_model.predict(X_test_tfidf)


# In[73]:


# import pandas as pd
# from IPython.display import FileLink

# # Load the sample submission file
# sub = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\SampleSubmission.csv")

# # Ensure all tweet_ids are present in predictions
# submission_with_predictions = pd.DataFrame({
#     'tweet_id': test_data['tweet_id'],
#     'location': test_predictions
# })

# # Merge the predictions with the sample submission to ensure all tweet_ids are covered
# final_submission = pd.merge(sub[['tweet_id']], submission_with_predictions, on='tweet_id', how='left')

# # Replace empty or whitespace-only locations with 'Unknown'
# final_submission['location'] = final_submission['location'].str.strip()
# final_submission['location'].replace('', 'Unknown', inplace=True)

# # Save the final submission file
# submission_file_path = r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\logistic_reg_submission_1.csv"
# final_submission.to_csv(submission_file_path, index=False)

# # Verify if the file is created successfully
# print(f"Submission file saved at: {submission_file_path}")

# # Create a download link for the submission file (if needed for Jupyter notebooks)
# def create_submission(submission_name):
#     return FileLink(submission_name)

# # Generate the download link
# download_link = create_submission(submission_file_path)

# # Display the download link (only relevant in Jupyter notebooks)
# download_link


# In[89]:


# Load the datasets
import pandas as pd

# Load the sample submission file
sub = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\SampleSubmission.csv")

# Ensure all tweet_ids are present in predictions
submission_with_predictions = pd.DataFrame({
    'tweet_id': test_data['tweet_id'],
    'location': test_predictions
})

# Merge the predictions with the sample submission to ensure all tweet_ids are covered
final_submission = pd.merge(sub[['tweet_id']], submission_with_predictions, on='tweet_id', how='left')

# Trim whitespace and handle empty strings
final_submission['location'] = final_submission['location'].str.strip()  # Remove leading and trailing whitespace
final_submission['location'].replace('', 'Unknown', inplace=True)  # Replace empty strings with "Unknown"

# Save the final submission file
submission_file_path = r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\final_submission_corrected.csv"
final_submission.to_csv(submission_file_path, index=False)

# Verify if the file is created successfully
print(f"Submission file saved at: {submission_file_path}")

# Create a download link for the submission file (if needed for Jupyter notebooks)
from IPython.display import FileLink

def create_submission(submission_name):
    return FileLink(submission_name)

# Generate the download link
download_link = create_submission(submission_file_path)

# Display the download link (only relevant in Jupyter notebooks)
download_link


# In[91]:


# Check for empty or whitespace-only locations
empty_or_whitespace_locations = final_submission['location'].apply(lambda x: not x.strip()).sum()

# Display the count of empty or whitespace-only locations
print(f"\nNumber of empty or whitespace-only locations: {empty_or_whitespace_locations}\n")

# Display the count of null values
print("Null values locations:")
print(final_submission.isnull().sum())


# In[92]:


(validation_wer, classification_report_validation, f1_score_validation, submission_file_path)


# In[ ]:




