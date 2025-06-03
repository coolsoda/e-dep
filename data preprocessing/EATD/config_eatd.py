#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

EATD_BASE_PATH = 'F:/EATD-Corpus/EATD-Corpus' 

# Output directory for all processed EATD data
PROCESSED_DATA_OUTPUT_DIR = 'F:/EATD-Corpus/processed_data_eatd'

PROCESSED_EATD_JSON_FILENAME = 'processed_eatd_interviews_qa.json'

AUDIO_FEATURES_EATD_DIR_NAME = 'audio_features_xlsr53_eatd' 

PROCESSED_EATD_JSON_WITH_FEATURES_FILENAME = 'processed_eatd_interviews_qa_with_audio_features.json'

if not os.path.exists(PROCESSED_DATA_OUTPUT_DIR):
    try:
        os.makedirs(PROCESSED_DATA_OUTPUT_DIR)
        print(f"Created base processed data directory for EATD: {PROCESSED_DATA_OUTPUT_DIR}")
    except Exception as e:
        print(f"Error creating base processed data directory {PROCESSED_DATA_OUTPUT_DIR}: {e}")
elif not os.path.isdir(PROCESSED_DATA_OUTPUT_DIR):
    print(f"Error: {PROCESSED_DATA_OUTPUT_DIR} exists but is not a directory.")

if not os.path.isdir(EATD_BASE_PATH):
    print(f"WARNING: The specified EATD_BASE_PATH does not exist or is not a directory: {EATD_BASE_PATH}")
    print("Please ensure the path in config_eatd.py is correct.")

