#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import os
from config_androids import METADATA_FILE_PATH, HC_AUDIO_PATH, PT_AUDIO_PATH

def load_and_structure_timedata():

    try:
        df = pd.read_csv(METADATA_FILE_PATH, header=None)
    except FileNotFoundError:
        print(f"Error: Metadata file not found at {METADATA_FILE_PATH}")
        return {}
    except Exception as e:
        print(f"Error reading metadata CSV {METADATA_FILE_PATH}: {e}")
        return {}

    participant_data = {}
    for index, row in df.iterrows():
        participant_id = str(row.iloc[0])
    
        label = None
        full_audio_path = None
        
        hc_file = os.path.join(HC_AUDIO_PATH, f"{participant_id}.wav")
        pt_file = os.path.join(PT_AUDIO_PATH, f"{participant_id}.wav")

        if os.path.exists(hc_file):
            label = 0 # Healthy Control
            full_audio_path = hc_file
        elif os.path.exists(pt_file):
            label = 1 # Depressed Participant
            full_audio_path = pt_file
        else:
            print(f"Warning: Audio file for participant {participant_id} not found in HC or PT folders. Skipping.")
            continue
            
        turns = []
        current_time = 0.0
        
        time_points = [tp for tp in row.iloc[1:].dropna().astype(float)]

        if not time_points or len(time_points) % 2 != 0:
            
            print(f"Warning: Participant {participant_id} has an odd number or no time_points: {time_points}. Skipping turns for this participant.")
            participant_data[participant_id] = {
                'label': label,
                'full_audio_path': full_audio_path,
                'turns': []
            }
            continue

        for i in range(0, len(time_points), 2):
            p_start = time_points[i]
            p_end = time_points[i+1]

            if p_start > current_time: # Ensure there's a gap for the interviewer
                turns.append({'speaker': 'Interviewer', 'start_time': current_time, 'stop_time': p_start})
            
            # Participant segment (A)
            turns.append({'speaker': 'Participant', 'start_time': p_start, 'stop_time': p_end})
            
            current_time = p_end # Update current time to the end of participant's speech
            
        # Potentially, there's a final interviewer utterance after the last participant segment
        # This is harder to determine without knowing the total audio duration from here.
        # We handle this by segmenting up to the last participant 'stop_time'.
        # For now, we only explicitly define turns based on the provided participant timestamps.

        participant_data[participant_id] = {
            'label': label,
            'full_audio_path': full_audio_path,
            'turns': turns
        }
        
    print(f"Loaded and structured timedata for {len(participant_data)} participants.")
    return participant_data

if __name__ == '__main__':
    # Example usage for testing this script
    structured_data = load_and_structure_timedata()
    if structured_data:
        # Print data for one example participant if they exist
        example_pid = '01_CF56_1'
        if example_pid in structured_data:
            print(f"\nData for participant {example_pid}:")
            print(f"  Label: {'PT' if structured_data[example_pid]['label'] == 1 else 'HC'}")
            print(f"  Audio Path: {structured_data[example_pid]['full_audio_path']}")
            print(f"  Turns (first 5):")
            for turn in structured_data[example_pid]['turns'][:5]:
                print(f"    {turn}")
        else:
            print(f"Example participant {example_pid} not found in structured data.")
            if len(structured_data) > 0:
                 print("\nFirst available participant data:")
                 first_pid = list(structured_data.keys())[0]
                 print(f"  Label: {'PT' if structured_data[first_pid]['label'] == 1 else 'HC'}")
                 print(f"  Audio Path: {structured_data[first_pid]['full_audio_path']}")
                 print(f"  Turns (first 5):")
                 for turn in structured_data[first_pid]['turns'][:5]:
                    print(f"    {turn}")
    else:
        print("Failed to load or structure timedata.")

