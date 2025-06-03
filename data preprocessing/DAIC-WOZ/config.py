#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

DAIC_WOZ_BASE_PATH = 'F:/DAIC-WOZ'

# Output directories
PROCESSED_DATA_DIR = os.path.join(DAIC_WOZ_BASE_PATH, 'processed_data')
AUDIO_SEGMENTS_DIR_NAME = 'audio_segments' 

LABELS_FILE_NAME = 'depression_levels_all.csv'
PROCESSED_JSON_FILENAME = 'processed_daic_woz_interviews_qa.json'


if not os.path.exists(PROCESSED_DATA_DIR):
    try:
        os.makedirs(PROCESSED_DATA_DIR)
        print(f"Created base processed data directory: {PROCESSED_DATA_DIR}")
    except Exception as e:
        print(f"Error creating base processed data directory {PROCESSED_DATA_DIR}: {e}")
elif not os.path.isdir(PROCESSED_DATA_DIR):
    print(f"Error: {PROCESSED_DATA_DIR} exists but is not a directory.")


if not os.path.isdir(DAIC_WOZ_BASE_PATH):
    print(f"WARNING: The specified DAIC_WOZ_BASE_PATH does not exist or is not a directory: {DAIC_WOZ_BASE_PATH}")
    print("Please ensure the path in config.py is correct.")

