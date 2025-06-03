#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
from config_eatd import EATD_BASE_PATH, PROCESSED_DATA_OUTPUT_DIR, PROCESSED_EATD_JSON_FILENAME
from load_labels_eatd import load_sds_score
from transcript_parser_eatd import parse_eatd_qa_file

def preprocess_eatd_dataset():
    print(f"Starting EATD preprocessing from base path: {EATD_BASE_PATH}")
    
    all_processed_interviews = []
    
    try:
        participant_folders = [d for d in os.listdir(EATD_BASE_PATH)
                               if os.path.isdir(os.path.join(EATD_BASE_PATH, d)) and d.startswith('t_')]
    except FileNotFoundError:
        print(f"Error: Base EATD dataset path {EATD_BASE_PATH} not found. Check config_eatd.py.")
        return None

    if not participant_folders:
        print(f"No participant folders (e.g., 't_1') found in {EATD_BASE_PATH}.")
        return None
        
    conditions = ["negative", "positive", "neutral"]

    for folder_name in participant_folders:
        participant_id_str = folder_name 
        participant_folder_path = os.path.join(EATD_BASE_PATH, folder_name)

        print(f"Processing participant: {participant_id_str}...")
        
        label_data = load_sds_score(participant_folder_path)
        if not label_data:
            print(f"Warning: Could not load labels for {participant_id_str}. Skipping.")
            continue
            
        participant_qa_pairs = []
        for condition in conditions:
            txt_file_path = os.path.join(participant_folder_path, f"{condition}.txt")
            audio_file_path = os.path.join(participant_folder_path, f"{condition}.wav") # Path to the audio file

            qa_content = parse_eatd_qa_file(txt_file_path)
            
            if qa_content:
                relative_audio_path = os.path.join(folder_name, f"{condition}.wav")
                absolute_audio_path = os.path.join(EATD_BASE_PATH, relative_audio_path)

                if not os.path.exists(absolute_audio_path):
                    print(f"Warning: Audio file {absolute_audio_path} not found for P:{participant_id_str} Cond:{condition}")

                qa_pair = {
                    'condition_type': condition,
                    'question_text': qa_content['question'],
                    'answer_text': qa_content['answer'],
                    'answer_audio_path_relative_to_dataset_base': relative_audio_path,
                    'answer_audio_path_absolute': absolute_audio_path
                }
                participant_qa_pairs.append(qa_pair)
            else:
                print(f"Warning: Could not parse {txt_file_path} for participant {participant_id_str}. Skipping this QA.")

        if not participant_qa_pairs and len(conditions) > 0 :
            print(f"Warning: No QA pairs successfully processed for {participant_id_str}. Skipping participant.")
            continue

        interview_data = {
            'participant_id': participant_id_str,
            'sds_score': label_data['sds_score'],
            'binary_label': label_data['binary_label'],
            'qa_pairs': participant_qa_pairs
        }
        all_processed_interviews.append(interview_data)
        print(f"Finished processing for participant {participant_id_str}. Processed {len(participant_qa_pairs)} QA pairs.")

    print(f"\nSuccessfully processed {len(all_processed_interviews)} interviews out of {len(participant_folders)} discoverable participant folders.")
    return all_processed_interviews

# --- Main Execution ---
if __name__ == '__main__':
    processed_data = preprocess_eatd_dataset()

    if processed_data:
        output_file_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, PROCESSED_EATD_JSON_FILENAME)
        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            print(f"\nProcessed EATD data saved to {output_file_path}")
            if len(processed_data) > 0:
                 print(f"\nSample of processed EATD data (first interview, if available):")
                 print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
            else:
                print("\nNo EATD interviews were processed successfully to show a sample.")
        except Exception as e:
            print(f"Error saving processed EATD data to {output_file_path}: {e}")
    else:
        print("\nEATD preprocessing failed or produced no data.")

