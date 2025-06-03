#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import time
from config_androids import (
    PROCESSED_DATA_ANDROID_DIR, 
    SEGMENTED_AUDIO_DIR_NAME, 
    PROCESSED_ANDROID_JSON_FILENAME
)
from metadata_parser_androids import load_and_structure_timedata
from audio_tools_androids import initialize_asr_model, segment_and_save_turn_audio, transcribe_audio_segment
from qa_structurer_androids import structure_qa_from_transcribed_turns

def preprocess_androids_dataset():
    
    print("Starting Androids dataset preprocessing...")
    
    asr_model, asr_processor = initialize_asr_model()
    if not asr_model:
        print("Failed to initialize ASR model. Cannot proceed with transcription. Exiting.")
        return None

    all_participant_metadata = load_and_structure_timedata()
    if not all_participant_metadata:
        print("Could not load participant metadata and turn structures. Exiting.")
        return None

    all_processed_interviews = []
    
    # Base directory for saving all segmented turns for this dataset
    base_segmented_audio_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, SEGMENTED_AUDIO_DIR_NAME)
    os.makedirs(base_segmented_audio_path, exist_ok=True)

    total_participants = len(all_participant_metadata)
    for current_idx, (participant_id, data) in enumerate(all_participant_metadata.items()):
        print(f"\nProcessing participant: {participant_id} ({current_idx + 1}/{total_participants})...")
        
        full_audio_path = data['full_audio_path']
        original_turns = data['turns'] # List of {'speaker', 'start_time', 'stop_time'}
        
        if not os.path.exists(full_audio_path):
            print(f"Warning: Full audio file {full_audio_path} for participant {participant_id} not found. Skipping.")
            continue
        if not original_turns:
            print(f"Warning: No turn data for participant {participant_id}. Skipping.")
            continue

        # Directory for this participant's segmented turns
        participant_segment_dir = os.path.join(base_segmented_audio_path, participant_id)
        os.makedirs(participant_segment_dir, exist_ok=True)
        
        transcribed_turns_for_interview = []
        total_turns_for_participant = len(original_turns)
        print(f"  Found {total_turns_for_participant} turns to process for {participant_id}.")

        for turn_idx, turn_info in enumerate(original_turns):
            print(f"    Processing turn {turn_idx + 1}/{total_turns_for_participant} ({turn_info['speaker']})...")
            
            segment_filename_base = f"{participant_id}_turn_{turn_idx+1}_{turn_info['speaker']}"
            
            absolute_segment_path = segment_and_save_turn_audio(
                full_audio_path, 
                turn_info, 
                turn_idx + 1,
                participant_segment_dir
            )
            
            turn_data_for_json = {
                'speaker': turn_info['speaker'],
                'start_time': turn_info['start_time'],
                'stop_time': turn_info['stop_time'],
                'segmented_audio_path_relative': None, 
                'text': "AUDIO_SEGMENTATION_FAILED" 
            }

            if absolute_segment_path:
                turn_data_for_json['segmented_audio_path_relative'] = os.path.join(
                    SEGMENTED_AUDIO_DIR_NAME, 
                    participant_id, 
                    os.path.basename(absolute_segment_path)
                )
                
                print(f"      Transcribing segment: {absolute_segment_path}...")
                start_transcription_time = time.time()
                transcription = transcribe_audio_segment(absolute_segment_path)
                end_transcription_time = time.time()
                print(f"      Transcription: '{transcription}' (took {end_transcription_time - start_transcription_time:.2f}s)")
                turn_data_for_json['text'] = transcription
            else:
                print(f"      Skipping transcription for turn {turn_idx+1} due to segmentation failure or invalid timestamps.")
            
            transcribed_turns_for_interview.append(turn_data_for_json)

        if not transcribed_turns_for_interview:
            print(f"Warning: No turns were successfully segmented/transcribed for {participant_id}. Skipping QA structuring.")
        
        qa_pairs = structure_qa_from_transcribed_turns(transcribed_turns_for_interview)
        
        interview_data_to_save = {
            'participant_id': participant_id,
            'depression_label': data['label'], # 0 for HC, 1 for PT
            'original_full_audio_path': data['full_audio_path'],
            'qa_pairs': qa_pairs,
        }
        all_processed_interviews.append(interview_data_to_save)
        print(f"  Finished participant {participant_id}. Extracted {len(qa_pairs)} QA pairs.")

    print(f"\nSuccessfully processed data for {len(all_processed_interviews)} participants.")
    return all_processed_interviews

# --- Main Execution ---
if __name__ == '__main__':
    overall_start_time = time.time()
    processed_data = preprocess_androids_dataset()
    overall_end_time = time.time()
    print(f"\nTotal Androids preprocessing time: {(overall_end_time - overall_start_time)/60:.2f} minutes.")

    if processed_data:
        output_file_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, PROCESSED_ANDROID_JSON_FILENAME)
        try:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)
            print(f"\nProcessed Androids data saved to {output_file_path}")
            if len(processed_data) > 0:
                 print(f"\nSample of processed Androids data (first participant, if available):")
                 print(json.dumps(processed_data[0], indent=2, ensure_ascii=False))
            else:
                print("\nNo Androids interviews were processed successfully to show a sample.")
        except Exception as e:
            print(f"Error saving processed Androids data to {output_file_path}: {e}")
    else:
        print("\nAndroids preprocessing failed or produced no data.")

