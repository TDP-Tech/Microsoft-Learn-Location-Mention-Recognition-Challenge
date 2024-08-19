#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING LIBRARIES

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ### LOADING DATA

# In[2]:


train_data = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\Train.csv")
test_data = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\Test.csv")
sample_submission = pd.read_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\SampleSubmission.csv")


# ### EXPLORING DATA

# In[3]:


print(train_data.head())


# In[4]:


print(test_data.head())


# In[5]:


print(sample_submission.head())


# ### UNDERSTANDING DATA

# **Train.csv:** This file should contain the microblogging posts along with the corresponding location mentions.
# 
# **Test.csv:** This file will have the microblogging posts but without the location mentions, which you need to predict.
# 
# **SampleSubmission.csv:** This shows the format in which you need to submit your predictions.

# ### BASIC STATISTICS

# In[6]:


print("\nTraining data statistics\n")
print(train_data.describe())


# In[7]:


# Check the columns in the dataset
print("\nColumns in Train.csv:")
print(train_data.columns)


# In[8]:


# Display basic information about the dataset
print("\nBasic Information:")
print(train_data.info())


# In[9]:


# check the number of unique locations in the training data
print("\nNumber of unique locations in the training data")
print(train_data['location'].unique())


# ### CHECKING FOR MISSING VALUES

# In[10]:


print("\nLength of a training data:\n")
print(len(train_data))
print("\nMissing values in training data:\n")
print(train_data.isnull().sum())


# ### TEXT PREPROCESSING

# Text preprocessing is a crucial part to ensure the model can effectively understand the content.

# In[11]:


# convert text to lower case
train_data['text'] = train_data['text'].str.lower()


# In[12]:


# Ensure all text entries are strings
train_data['text'] = train_data['text'].astype(str)


# In[13]:


# fill NaN values with empty string
train_data['text'] = train_data['text'].fillna('')


# In[14]:


# remove unnecessary symbols
import re

#remove urls, mentions, hashtags, and special characters
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'http\S+|www\S+|@\S+|#\S+', '', x))
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x)) # remove punctuation
train_data['text'] = train_data['text'].apply(lambda x: re.sub(r'\d+', '', x)) # remove numbers


# ### TOKENIZATION

# Tokenization involves breaking down the text into individual words (tokens). 
# 
# This step is essential for further processing and model training.

# In[15]:


import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')


# In[16]:


# Tokenize the text
train_data['tokens'] = train_data['text'].apply(word_tokenize)


# In[17]:


# Display a few examples of tokenized text
print('\nTokenized text examples')
print(train_data[['text', 'tokens']].head())


# ### HANDLE ABBREVIATIONS AND COMMON TERMS

# Location mentions might include abbreviations, for example "NYC" for New York City, "NY" for New York and so on.
# 
# Handling these abbreviations is crucial for accurate location recognition.

# In[18]:


# create a dictionary for handling abbreviations

abbreviations = {
    "nyc": "new york city",
    "la": "los angeles",
    "sf": "san ransisco",
    "dc": "washington dc",
}


# In[19]:


def replace_abbreviations(text, abbreviation_dict):
    words = text.split()
    replaced_text = ' '.join([abbreviation_dict.get(word, word) for word in words])
    return replaced_text

train_data['text'] = train_data['text'].apply(lambda x: replace_abbreviations(x, abbreviations))


# In[20]:


# Display the modified text after abbreviation replaced

print("\nText after handling abbreviations:")
print(train_data['text'].head())


# ### SAVE THE PROCESSED MODEL

# In[21]:


# Define the file path to save the preprocessed data
processed_file_path = r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\Processed_Train.csv"


# In[22]:


# Save the processed data to a new CSV file
train_data.to_csv(processed_file_path, index=False)


# In[23]:


# Display the first five rows of the processed data
print("\nProcessed Train Data")
print(train_data.head())


# In[24]:


print(f"Processed data saved to {processed_file_path}")


# ### Step 1: Vectorizing the Text Data using TF-IDF

# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # adjust according to your needs

# Fit and Transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['text'])

# Transform the test data
X_test_tfidf = tfidf_vectorizer.transform(test_data['text'])


# In[26]:


# Display the shape of the resulting matrices
print("TF-IDF matrix for training data:", X_train_tfidf.shape)
print("TF-IDF matrix for test data:", X_test_tfidf.shape)


# ### Step 2.1: Process the target variable

# In[27]:


from sklearn.preprocessing import MultiLabelBinarizer
# Fill NaN values with an empty string
train_data['location'] = train_data['location'].fillna('')

# Split the location mention into a list
train_data['location_list'] = train_data['location'].apply(lambda x: x.split())

# Initialize the MultiLabelBinarizer
mlb = MultiLabelBinarizer()

# Fit and transform the location list to a binary matrix
y_train = mlb.fit_transform(train_data['location_list'])


# In[28]:


# Display the shape of the resulting matrix
print("Shape of the target variable matrix:", y_train.shape)


# ### MODEL SELECTION AND TRAINING

# In[30]:


# LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Initialize the logistic regression model
lr_model = LogisticRegression(solver='liblinear')

# Use OneVsRestClassifier for multi-label classification
model = OneVsRestClassifier(lr_model)

# Train the model
model.fit(X_train_tfidf, y_train)


# In[31]:


# Display the training accuracy
training_accuracy = model.score(X_train_tfidf, y_train)
print("Training accuracy: {:.2f}".format(training_accuracy * 100))


# ### Step 3.2: Predicting on Test Data
# 
# Predict the locations after training the model

# In[ ]:


# Predict the locations for the teX_test_tfidfst_tfidfst_tfidfst_tfidfst_tfidfst_tfidfst_tfidfst_tfidfst_tfidfata
y_test_prediction = model.predict(X_test_tfidf)

# Convert the preinverse_transform back to the original location format
predicted_locations = mlb.inverse_transform(y_test_prediction)


# In[ ]:


# Prepair the prediction file for submission
submission_df = test_data[['ID']].copy()
submission_dfission_dfission_df['locations'] = [' '.join(loc) for loc in predicted_locations]


# In[ ]:


# Save the submission file
submission_df.to_csv(r"E:\AI,ML,NLP PROJECTS\Microsoft Learn Location Mention Recognition Challenge\Submission.csv", index=False)
print("Submission file created and saved successfully!")

