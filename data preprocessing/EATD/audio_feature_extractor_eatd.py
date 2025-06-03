#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import librosa
import os
import numpy as np
import json
from config_eatd import PROCESSED_DATA_OUTPUT_DIR, PROCESSED_EATD_JSON_FILENAME,                         AUDIO_FEATURES_EATD_DIR_NAME, PROCESSED_EATD_JSON_WITH_FEATURES_FILENAME

def initialize_audio_model_and_processor(model_name="facebook/wav2vec2-large-xlsr-53"):
    """Initializes and returns the Wav2Vec2 model and feature extractor."""
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()
        print(f"Successfully loaded model and feature extractor for {model_name}")
        return model, feature_extractor
    except Exception as e:
        print(f"Error loading Wav2Vec2 model/feature_extractor: {e}")
        return None, None

def extract_features_for_segment(audio_path, model, feature_extractor, target_sr=16000):
    
    if not audio_path or not os.path.exists(audio_path):
        print(f"Warning: Audio segment not found or path is invalid: '{audio_path}'. Skipping feature extraction.")
        return None
        
    try:
        waveform, sr = librosa.load(audio_path, sr=None) 
        if sr != target_sr:
            waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        input_values = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True).input_values

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_values = input_values.to(device)
        model.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            last_hidden_state = outputs.last_hidden_state 
        
        pooled_features = torch.mean(last_hidden_state.squeeze(0), dim=0)
        
        return pooled_features.cpu().numpy()

    except Exception as e:
        print(f"Error extracting features for {audio_path}: {e}")
        return None

def process_all_eatd_audio_features():
    
    model, feature_extractor = initialize_audio_model_and_processor()
    if model is None or feature_extractor is None:
        print("Failed to initialize audio model. Cannot extract features.")
        return

    input_json_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, PROCESSED_EATD_JSON_FILENAME)
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            all_interviews_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed EATD JSON file not found at {input_json_path}")
        print("Please run main_preprocess_eatd.py first.")
        return
    except Exception as e:
        print(f"Error reading EATD JSON {input_json_path}: {e}")
        return
        
    base_output_features_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, AUDIO_FEATURES_EATD_DIR_NAME)
    os.makedirs(base_output_features_path, exist_ok=True)
    print(f"Saving extracted EATD audio features to: {base_output_features_path}")

    updated_interviews_data = []

    for interview_idx, interview_data in enumerate(all_interviews_data):
        participant_id = interview_data['participant_id'] # e.g., "t_1"
        print(f"Extracting audio features for EATD participant {participant_id} ({interview_idx+1}/{len(all_interviews_data)})...")
        
        participant_feature_dir = os.path.join(base_output_features_path, participant_id)
        os.makedirs(participant_feature_dir, exist_ok=True)
        
        updated_qa_pairs = []
        for qa_pair in interview_data['qa_pairs']:
            absolute_audio_path = qa_pair.get('answer_audio_path_absolute')
            condition_type = qa_pair.get('condition_type', 'unknown_condition')
            
            feature_vector = None
            feature_vector_path_relative = None

            if absolute_audio_path: # Check if path exists
                feature_vector = extract_features_for_segment(absolute_audio_path, model, feature_extractor)
                if feature_vector is not None:
                    feature_filename = f"{participant_id}_{condition_type}_answer_xlsr53.npy"
                    absolute_feature_path = os.path.join(participant_feature_dir, feature_filename)
                    np.save(absolute_feature_path, feature_vector)

                    feature_vector_path_relative = os.path.join(AUDIO_FEATURES_EATD_DIR_NAME, participant_id, feature_filename)
            else:
                if qa_pair.get('answer_text'):
                    print(f"Warning: No absolute audio path for P:{participant_id} Condition:{condition_type}. Skipping feature extraction.")

            qa_pair['audio_features_xlsr53_path_relative'] = feature_vector_path_relative
            updated_qa_pairs.append(qa_pair)
        
        interview_data['qa_pairs'] = updated_qa_pairs
        updated_interviews_data.append(interview_data)

    # Save the updated JSON file with paths to the feature vectors
    updated_json_path = os.path.join(PROCESSED_DATA_OUTPUT_DIR, PROCESSED_EATD_JSON_WITH_FEATURES_FILENAME)
    try:
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_interviews_data, f, indent=4, ensure_ascii=False)
        print(f"\nUpdated EATD data with audio feature paths saved to {updated_json_path}")
    except Exception as e:
        print(f"Error saving updated EATD JSON data: {e}")

if __name__ == '__main__':
    # This script assumes that `main_preprocess_eatd.py` has already been run
    # and `processed_eatd_interviews_qa.json` exists in PROCESSED_DATA_OUTPUT_DIR
    # and the paths `answer_audio_path_absolute` in it are correct.
    
    input_json_file = os.path.join(PROCESSED_DATA_OUTPUT_DIR, PROCESSED_EATD_JSON_FILENAME)
    
    if not os.path.exists(input_json_file):
        print(f"Error: Input JSON file for EATD not found at {input_json_file}")
        print("Please run the main_preprocess_eatd.py script first to generate the initial JSON.")
    else:
        print("Starting EATD audio feature extraction process...")
        process_all_eatd_audio_features()
        print("EATD audio feature extraction process complete.")

