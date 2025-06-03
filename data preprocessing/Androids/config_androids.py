#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os

ANDROIDS_CORPUS_BASE_PATH = 'F:/Androids-Corpus/Androids-Corpus'
AUDIO_ROOT_PATH = os.path.join(ANDROIDS_CORPUS_BASE_PATH, 'Interview-Task', 'audio')
HC_AUDIO_PATH = os.path.join(AUDIO_ROOT_PATH, 'HC')
PT_AUDIO_PATH = os.path.join(AUDIO_ROOT_PATH, 'PT')
METADATA_FILE_PATH = os.path.join(ANDROIDS_CORPUS_BASE_PATH, 'interview_timedata.csv')


PROCESSED_DATA_ANDROID_DIR = os.path.join(ANDROIDS_CORPUS_BASE_PATH, 'processed_data_androids')
SEGMENTED_AUDIO_DIR_NAME = 'audio_segments_per_turn' 
# JSON output from main_preprocess_androids.py (contains ASR text and paths to segmented audio)
JSON_FROM_ASR_AND_SEGMENTATION = 'processed_androids_interviews_qa.json' 


AUDIO_FEATURES_ANDROID_DIR_NAME = 'audio_features_xlsr53_androids' 

INPUT_JSON_FROM_AUDIO_EXTRACTION_ANDROID = 'processed_androids_interviews_qa_with_audio_features.json' 


TEXT_FEATURES_ANDROID_DIR_NAME = 'text_features_xlmr_androids' 

PROCESSED_ANDROID_JSON_WITH_ALL_FEATURES_FILENAME = 'processed_androids_interviews_qa_with_all_features.json'


ASR_MODEL_NAME = "openai/whisper-base" 
ASR_LANGUAGE = "italian" 


XLM_R_MODEL_NAME = "xlm-roberta-base" 
XLM_R_MAX_LENGTH = 256 


if not os.path.exists(PROCESSED_DATA_ANDROID_DIR):
    try:
        os.makedirs(PROCESSED_DATA_ANDROID_DIR)
        print(f"Created base processed data directory for Androids: {PROCESSED_DATA_ANDROID_DIR}")
    except Exception as e:
        print(f"Error creating base processed data directory {PROCESSED_DATA_ANDROID_DIR}: {e}")
elif not os.path.isdir(PROCESSED_DATA_ANDROID_DIR):
    print(f"Error: {PROCESSED_DATA_ANDROID_DIR} exists but is not a directory.")


for path_to_check in [ANDROIDS_CORPUS_BASE_PATH, AUDIO_ROOT_PATH, HC_AUDIO_PATH, PT_AUDIO_PATH, METADATA_FILE_PATH]:
    if not os.path.exists(path_to_check):
        print(f"WARNING: Required input path does not exist: {path_to_check}")
        print("Please ensure the paths in config_androids.py are correct.")
