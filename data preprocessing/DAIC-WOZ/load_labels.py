#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from config import DAIC_WOZ_BASE_PATH, LABELS_FILE_NAME

def load_depression_labels():
    """
    Loads depression labels from the CSV file specified in config.py.

    """
    file_path = os.path.join(DAIC_WOZ_BASE_PATH, LABELS_FILE_NAME)
    try:
        labels_df = pd.read_csv(file_path)
        labels_df['Participant_ID'] = labels_df['Participant_ID'].astype(str) 
        
        participant_labels = {}
        for _, row in labels_df.iterrows():
            pid = row['Participant_ID']
            phq8_score = row['PHQ8_Score']
            binary_label = 1 if phq8_score >= 10 else 0
            participant_labels[pid] = {
                'phq8_score': phq8_score,
                'binary_label': binary_label,
            }
        print(f"Successfully loaded labels for {len(participant_labels)} participants from {file_path}.")
        return participant_labels
    except FileNotFoundError:
        print(f"Error: Label file not found at {file_path}")
        return {}
    except Exception as e:
        print(f"Error loading labels from {file_path}: {e}")
        return {}

if __name__ == '__main__':
    # Example usage for testing this script independently
    labels = load_depression_labels()
    if labels:
        if '300' in labels: 
            print(f"Labels for participant 300: {labels['300']}")
        else:
            print("Labels loaded, but example participant '300' not found.")
    else:
        print("Failed to load labels for testing.")

