#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

def load_sds_score(participant_folder_path):
    """
    Loads SDS score from the label.txt file in a participant's folder.

    Args:
        participant_folder_path (str): Absolute path to the participant's folder (e.g., 'F:/EATD-Corpus/EATD-Corpus/t_1').

    Returns:
        dict: {'sds_score': float, 'binary_label': int (0 for healthy, 1 for depressed)}
              Returns None if file not found or error in parsing.
    """
    label_file_path = os.path.join(participant_folder_path, 'label.txt')
    try:
        with open(label_file_path, 'r') as f:
            sds_score_str = f.readline().strip()
            sds_score = float(sds_score_str)
        
        binary_label = 1 if sds_score >= 63 else 0
        return {
            'sds_score': sds_score,
            'binary_label': binary_label
        }
    except FileNotFoundError:
        print(f"Error: label.txt not found in {participant_folder_path}")
        return None
    except ValueError:
        print(f"Error: Could not convert SDS score '{sds_score_str}' to float in {label_file_path}")
        return None
    except Exception as e:
        print(f"Error loading SDS score from {label_file_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this script
    from config_eatd import EATD_BASE_PATH 
    
    example_participant_folder = os.path.join(EATD_BASE_PATH, 't_1')
    if os.path.isdir(example_participant_folder):
        labels = load_sds_score(example_participant_folder)
        if labels:
            print(f"Labels for participant {os.path.basename(example_participant_folder)}: {labels}")
        else:
            print(f"Failed to load labels for {os.path.basename(example_participant_folder)}.")
    else:
        print(f"Example participant folder not found: {example_participant_folder}")
        print("Please ensure EATD_BASE_PATH in config_eatd.py is correct and participant folders like 't_1' exist.")

