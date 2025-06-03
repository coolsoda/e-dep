#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
from config import DAIC_WOZ_BASE_PATH, PROCESSED_DATA_DIR, AUDIO_SEGMENTS_DIR_NAME, PROCESSED_JSON_FILENAME
from load_labels import load_depression_labels       
from transcript_parser import process_transcript_file 
from qa_extractor import extract_qa_pairs             
from audio_processor import segment_and_save_audio    

def preprocess_daic_woz_dataset():

    print(f"Starting DAIC-WOZ preprocessing from base path: {DAIC_WOZ_BASE_PATH}")
    
    all_participant_labels = load_depression_labels()
    if not all_participant_labels:
        print("Could not load participant labels. Exiting.")
        return None

    all_processed_interviews = []
    
    try:
        participant_folders = [d for d in os.listdir(DAIC_WOZ_BASE_PATH) 
                               if os.path.isdir(os.path.join(DAIC_WOZ_BASE_PATH, d)) and d.endswith('_P')]
    except FileNotFoundError:
        print(f"Error: Base dataset path {DAIC_WOZ_BASE_PATH} not found. Check config.py.")
        return None
    
    if not participant_folders:
        print(f"No participant folders (e.g., '300_P') found in {DAIC_WOZ_BASE_PATH}.")
        return None

    # Create base directory for all audio segments if it doesn't exist
    base_audio_segments_path = os.path.join(PROCESSED_DATA_DIR, AUDIO_SEGMENTS_DIR_NAME)
    os.makedirs(base_audio_segments_path, exist_ok=True)

    for folder_name in participant_folders: # e.g., "300_P"
        participant_id_str = folder_name.split('_')[0] # e.g., "300"
        
        if participant_id_str not in all_participant_labels:
            print(f"Warning: No label found for participant {participant_id_str}. Skipping.")
            continue

        print(f"Processing participant: {participant_id_str}...")
        
        transcript_file_path = os.path.join(DAIC_WOZ_BASE_PATH, folder_name, f"{participant_id_str}_TRANSCRIPT.csv")
        full_audio_file_path = os.path.join(DAIC_WOZ_BASE_PATH, folder_name, f"{participant_id_str}_AUDIO.wav") # Path for full audio

        if not os.path.exists(transcript_file_path):
            print(f"Warning: Transcript file not found for {participant_id_str}. Skipping.")
            continue
        if not os.path.exists(full_audio_file_path):
            print(f"Warning: Full audio file not found for {participant_id_str} at {full_audio_file_path}. Audio segments will not be created.")

        structured_transcript = process_transcript_file(transcript_file_path)
        if not structured_transcript:
            print(f"Skipping participant {participant_id_str} due to transcript processing issues.")
            continue
            
        qa_pairs_from_text = extract_qa_pairs(structured_transcript)
        
        processed_qa_pairs_for_interview = []
        if qa_pairs_from_text:
            # Create participant-specific directory for their audio segments
            participant_segment_output_dir = os.path.join(base_audio_segments_path, folder_name)
            os.makedirs(participant_segment_output_dir, exist_ok=True)

            for idx, qa in enumerate(qa_pairs_from_text):
                qa_turn_id = idx + 1
                answer_audio_segment_path = None
                relative_segment_path = None

                if os.path.exists(full_audio_file_path) and                    qa.get('answer_start_time') is not None and                    qa.get('answer_stop_time') is not None:
                    
                    segment_filename = f"{participant_id_str}_turn_{qa_turn_id}_answer.wav"
                    absolute_segment_path = os.path.join(participant_segment_output_dir, segment_filename)
                    
                    if segment_and_save_audio(full_audio_file_path, 
                                              qa['answer_start_time'], 
                                              qa['answer_stop_time'], 
                                              absolute_segment_path):
                        answer_audio_segment_path = absolute_segment_path

                        relative_segment_path = os.path.join(AUDIO_SEGMENTS_DIR_NAME, folder_name, segment_filename)

                # Update the qa dictionary
                qa['answer_audio_segment_path_absolute'] = answer_audio_segment_path
                qa['answer_audio_segment_path_relative'] = relative_segment_path # Store relative to PROCESSED_DATA_DIR
                processed_qa_pairs_for_interview.append(qa)
        else:
            print(f"Warning: No QA pairs extracted for {participant_id_str}.")


        interview_data = {
            'participant_id': participant_id_str,
            'phq8_score': all_participant_labels[participant_id_str]['phq8_score'],
            'binary_label': all_participant_labels[participant_id_str]['binary_label'],
            'qa_pairs': processed_qa_pairs_for_interview,
            'original_transcript_path': os.path.join(folder_name, f"{participant_id_str}_TRANSCRIPT.csv"), # Relative to DAIC_WOZ_BASE_PATH
            'original_audio_path': os.path.join(folder_name, f"{participant_id_str}_AUDIO.wav") # Relative to DAIC_WOZ_BASE_PATH
        }
        all_processed_interviews.append(interview_data)
        print(f"Finished processing for participant {participant_id_str}. Processed {len(processed_qa_pairs_for_interview)} QA pairs.")

    print(f"\nSuccessfully processed {len(all_processed_interviews)} interviews out of {len(participant_folders)} discoverable participant folders.")
    return all_processed_interviews

# --- Main Execution ---
if __name__ == '__main__':
    processed_data = preprocess_daic_woz_dataset()

    if processed_data:
        output_file_path = os.path.join(PROCESSED_DATA_DIR, PROCESSED_JSON_FILENAME)
        try:
            # Ensure output directory exists for the JSON file itself
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w') as f:
                json.dump(processed_data, f, indent=4)
            print(f"\nProcessed data saved to {output_file_path}")
            if len(processed_data) > 0:
                 print(f"\nSample of processed data (first interview, if available):")
                 print(json.dumps(processed_data[0], indent=2))
            else:
                print("\nNo interviews were processed successfully to show a sample.")
        except Exception as e:
            print(f"Error saving processed data to {output_file_path}: {e}")
    else:
        print("\nPreprocessing failed or produced no data.")

