#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import librosa
import os
import numpy as np
import json
from config_androids import (
    PROCESSED_DATA_ANDROID_DIR, 
    PROCESSED_ANDROID_JSON_FILENAME,
    AUDIO_FEATURES_ANDROID_DIR_NAME, 
    PROCESSED_ANDROID_JSON_WITH_FEATURES_FILENAME
)

_XLSR_MODEL = None
_XLSR_FEATURE_EXTRACTOR = None

def initialize_audio_model_and_processor(model_name="facebook/wav2vec2-large-xlsr-53"):
    global _XLSR_MODEL, _XLSR_FEATURE_EXTRACTOR
    if _XLSR_MODEL is None or _XLSR_FEATURE_EXTRACTOR is None:
        try:
            _XLSR_FEATURE_EXTRACTOR = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            _XLSR_MODEL = Wav2Vec2Model.from_pretrained(model_name)
            _XLSR_MODEL.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _XLSR_MODEL.to(device)
            print(f"Successfully loaded XLSR-53 model and feature extractor to {device} for feature extraction.")
        except Exception as e:
            print(f"Error loading Wav2Vec2 model/feature_extractor: {e}")
            _XLSR_MODEL = None
            _XLSR_FEATURE_EXTRACTOR = None
    return _XLSR_MODEL, _XLSR_FEATURE_EXTRACTOR

def extract_features_for_segment(audio_path, model, feature_extractor, target_sr=16000):
    if not audio_path or not os.path.exists(audio_path):
        return None
        
    try:
        waveform, sr = librosa.load(audio_path, sr=None)
        if sr != target_sr:
            waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        input_values = feature_extractor(waveform, sampling_rate=sr, return_tensors="pt", padding=True).input_values

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_values = input_values.to(device)

        with torch.no_grad():
            outputs = model(input_values)
            last_hidden_state = outputs.last_hidden_state 
        
        pooled_features = torch.mean(last_hidden_state.squeeze(0), dim=0)
        return pooled_features.cpu().numpy()

    except Exception as e:
        print(f"Error extracting features for {audio_path}: {e}")
        return None

# Main
def process_all_androids_audio_features():
    xlsr_model, xlsr_feature_extractor = initialize_audio_model_and_processor()
    if xlsr_model is None or xlsr_feature_extractor is None:
        print("Failed to initialize XLSR-53 audio model. Cannot extract features.")
        return

    input_json_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, PROCESSED_ANDROID_JSON_FILENAME)
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            all_interviews_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed Androids JSON file not found at {input_json_path}")
        print("Please run main_preprocess_androids.py first to generate segmented audio and transcripts.")
        return
    except Exception as e:
        print(f"Error reading Androids JSON {input_json_path}: {e}")
        return
        
    base_output_features_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, AUDIO_FEATURES_ANDROID_DIR_NAME)
    os.makedirs(base_output_features_path, exist_ok=True)
    print(f"Saving extracted Androids audio features to: {base_output_features_path}")

    updated_interviews_data = []

    total_interviews = len(all_interviews_data)
    for interview_idx, interview_data in enumerate(all_interviews_data):
        participant_id = interview_data['participant_id']
        print(f"Extracting audio features for Androids participant {participant_id} ({interview_idx+1}/{total_interviews})...")
        
        # Directory for this participant's extracted features
        participant_feature_dir = os.path.join(base_output_features_path, participant_id)
        os.makedirs(participant_feature_dir, exist_ok=True)
        
        updated_qa_pairs = []
        for qa_idx, qa_pair in enumerate(interview_data['qa_pairs']):
            relative_audio_segment_path = qa_pair.get('answer_audio_path_relative')
            
            feature_vector = None
            feature_vector_path_relative_to_processed_dir = None

            if relative_audio_segment_path:
                absolute_audio_segment_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, relative_audio_segment_path)
                
                if os.path.exists(absolute_audio_segment_path):
                    feature_vector = extract_features_for_segment(absolute_audio_segment_path, xlsr_model, xlsr_feature_extractor)
                    if feature_vector is not None:
                        segment_basename = os.path.basename(absolute_audio_segment_path)
                        feature_filename = segment_basename.replace(".wav", "_xlsr53.npy") 
                
                        absolute_feature_path = os.path.join(participant_feature_dir, feature_filename)
                        np.save(absolute_feature_path, feature_vector)
                        
                        feature_vector_path_relative_to_processed_dir = os.path.join(
                            AUDIO_FEATURES_ANDROID_DIR_NAME, 
                            participant_id, 
                            feature_filename
                        )
                else:
                    if qa_pair.get('answer_text'):
                         print(f"Warning: Segmented audio file not found at {absolute_audio_segment_path} for P:{participant_id} QA:{qa_idx}. Skipping feature extraction.")
            else:
                if qa_pair.get('answer_text'):
                    print(f"Warning: No relative audio path for participant answer for P:{participant_id} QA:{qa_idx}. Skipping feature extraction.")

            qa_pair['participant_answer_audio_features_xlsr53_path_relative'] = feature_vector_path_relative_to_processed_dir
            updated_qa_pairs.append(qa_pair)
        
        interview_data['qa_pairs'] = updated_qa_pairs
        updated_interviews_data.append(interview_data)

    updated_json_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, PROCESSED_ANDROID_JSON_WITH_FEATURES_FILENAME)
    try:
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_interviews_data, f, indent=4, ensure_ascii=False)
        print(f"\nUpdated Androids data with audio feature paths saved to {updated_json_path}")
    except Exception as e:
        print(f"Error saving updated Androids JSON data: {e}")

if __name__ == '__main__':
    
    input_json_file = os.path.join(PROCESSED_DATA_ANDROID_DIR, PROCESSED_ANDROID_JSON_FILENAME)
    
    if not os.path.exists(input_json_file):
        print(f"Error: Input JSON file for Androids not found at {input_json_file}")
        print("Please run the main_preprocess_androids.py script first.")
    else:
        print("Starting Androids audio feature extraction process...")
        process_all_androids_audio_features()
        print("Androids audio feature extraction process complete.")

