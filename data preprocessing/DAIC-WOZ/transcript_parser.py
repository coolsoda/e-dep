#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re

def parse_transcript_row(row_string):
    """
    Parses a single concatenated row string from the transcript.
    
    """
    match = re.match(r"(\d+\.?\d*)(\d+\.?\d*)(Participant|Ellie)(.*)", str(row_string))
    if match:
        start_time = float(match.group(1))
        stop_time = float(match.group(2))
        speaker = match.group(3)
        value = match.group(4).strip()
        return {'start_time': start_time, 'stop_time': stop_time, 'speaker': speaker, 'value': value}
    else:
        speaker_val = None
        text_val = str(row_string)
        if "Ellie" in text_val:
            speaker_val = "Ellie"
            parts = text_val.split("Ellie", 1)
            text_val = parts[1].strip() if len(parts) > 1 else ""
        elif "Participant" in text_val:
            speaker_val = "Participant"
            parts = text_val.split("Participant", 1)
            text_val = parts[1].strip() if len(parts) > 1 else ""
        
        if speaker_val:
            return {'start_time': None, 'stop_time': None, 'speaker': speaker_val, 'value': text_val}
        else:
            print(f"Warning: Could not parse row comprehensively: '{row_string}'")
            return None

def process_transcript_file(transcript_file_path):
    """
    Reads and processes a single participant's transcript CSV file.

    Returns:
        list: A list of dictionaries, each representing a turn. Empty list on error.
    """
    structured_transcript = []
    try:
        transcript_df = pd.read_csv(transcript_file_path)
        if transcript_df.empty or len(transcript_df.columns) == 0:
            print(f"Warning: Transcript file {transcript_file_path} is empty or has no columns.")
            return []
        column_name = transcript_df.columns[0]

        for _, row_series in transcript_df.iterrows():
            concatenated_string = row_series[column_name]
            parsed_data = parse_transcript_row(concatenated_string)
            if parsed_data and parsed_data.get('value') and parsed_data['value'].lower() != '<no speech>':
                structured_transcript.append(parsed_data)
        return structured_transcript
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {transcript_file_path}")
        return []
    except pd.errors.EmptyDataError:
        print(f"Warning: Transcript file {transcript_file_path} is empty.")
        return []
    except Exception as e:
        print(f"Error processing transcript {transcript_file_path}: {e}")
        return []

if __name__ == '__main__':
    # Example usage for testing this script

    from config import DAIC_WOZ_BASE_PATH
    import os
    
    example_participant_id = '300' # Change as needed
    example_transcript_path = os.path.join(DAIC_WOZ_BASE_PATH, f"{example_participant_id}_P", f"{example_participant_id}_TRANSCRIPT.csv")
    
    if os.path.exists(example_transcript_path):
        parsed_transcript = process_transcript_file(example_transcript_path)
        if parsed_transcript:
            print(f"Processed transcript for participant {example_participant_id} (first 5 turns):")
            for turn in parsed_transcript[:5]:
                print(turn)
        else:
            print(f"Failed to process transcript {example_transcript_path} for testing.")
    else:
        print(f"Example transcript file not found at: {example_transcript_path}")
        print("Please create a dummy file or ensure DAIC_WOZ_BASE_PATH in config.py is correct and the file exists.")

