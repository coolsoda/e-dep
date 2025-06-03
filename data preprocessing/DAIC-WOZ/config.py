#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

DAIC_WOZ_BASE_PATH = 'F:/DAIC-WOZ'
PROCESSED_DATA_DIR = os.path.join(DAIC_WOZ_BASE_PATH, 'processed_data')

LABELS_FILE_NAME = 'depression_levels_all.csv'
AUDIO_SEGMENTS_DIR_NAME = 'audio_segments_daic_woz'
INPUT_JSON_FROM_AUDIO_EXTRACTION = 'processed_daic_woz_interviews_qa_with_audio_features.json' 

TEXT_FEATURES_DAIC_WOZ_DIR_NAME = 'text_features_xlmr_daic_woz'

# Final JSON output name for DAIC-WOZ after text feature extraction
PROCESSED_DAIC_WOZ_JSON_WITH_ALL_FEATURES_FILENAME = 'processed_daic_woz_interviews_qa_with_all_features.json'

XLM_R_MODEL_NAME = "xlm-roberta-base" 
XLM_R_MAX_LENGTH = 256


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
    print("Please ensure the path in config_daic_woz.py is correct.")
