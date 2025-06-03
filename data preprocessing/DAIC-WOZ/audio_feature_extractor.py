

import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor # Wav2Vec2Model for features
import librosa
import os
import numpy as np
import json
from config import PROCESSED_DATA_DIR, AUDIO_SEGMENTS_DIR_NAME, PROCESSED_JSON_FILENAME


from transformers import Wav2Vec2Model


def initialize_audio_model_and_processor(model_name="facebook/wav2vec2-large-xlsr-53"):
    try:
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2Model.from_pretrained(model_name)
        model.eval()
        print(f"Successfully loaded model and processor for {model_name}")
        return model, processor
    except Exception as e:
        print(f"Error loading Wav2Vec2 model/processor: {e}")
        return None, None

def extract_features_for_segment(audio_path, model, processor, target_sr=16000):
    if not os.path.exists(audio_path):
        print(f"Warning: Audio segment not found at {audio_path}. Skipping feature extraction.")
        return None
        
    try:
        waveform, sr = librosa.load(audio_path, sr=None)
        if sr != target_sr:
            waveform = librosa.resample(y=waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        input_values = processor(waveform, sampling_rate=sr, return_tensors="pt", padding=True).input_values

        # Move to GPU if available
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

def process_all_audio_features(processed_json_path, output_features_dir_name="audio_features_xlsr53"):

    model, processor = initialize_audio_model_and_processor()
    if model is None or processor is None:
        print("Failed to initialize audio model. Cannot extract features.")
        return

    try:
        with open(processed_json_path, 'r') as f:
            all_interviews_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Processed JSON file not found at {processed_json_path}")
        return
    except Exception as e:
        print(f"Error reading JSON {processed_json_path}: {e}")
        return
        
    base_output_features_path = os.path.join(PROCESSED_DATA_DIR, output_features_dir_name)
    os.makedirs(base_output_features_path, exist_ok=True)
    print(f"Saving extracted audio features to: {base_output_features_path}")

    updated_interviews_data = []

    for interview_idx, interview_data in enumerate(all_interviews_data):
        participant_id = interview_data['participant_id']
        print(f"Extracting audio features for participant {participant_id} ({interview_idx+1}/{len(all_interviews_data)})...")
        
        participant_feature_dir = os.path.join(base_output_features_path, f"{participant_id}_P")
        os.makedirs(participant_feature_dir, exist_ok=True)
        
        updated_qa_pairs = []
        for qa_idx, qa_pair in enumerate(interview_data['qa_pairs']):

            audio_segment_path = qa_pair.get('answer_audio_segment_path_absolute')
            
            if not audio_segment_path and qa_pair.get('answer_audio_segment_path_relative'):
                audio_segment_path = os.path.join(PROCESSED_DATA_DIR, qa_pair['answer_audio_segment_path_relative'])

            feature_vector = None
            feature_vector_path_relative = None

            if audio_segment_path and os.path.exists(audio_segment_path):
                feature_vector = extract_features_for_segment(audio_segment_path, model, processor)
                if feature_vector is not None:
                    # Save the feature vector as a .npy file
                    feature_filename = f"{participant_id}_turn_{qa_idx+1}_answer_xlsr53.npy"
                    absolute_feature_path = os.path.join(participant_feature_dir, feature_filename)
                    np.save(absolute_feature_path, feature_vector)
                    feature_vector_path_relative = os.path.join(output_features_dir_name, f"{participant_id}_P", feature_filename)
            else:
                if qa_pair.get('answer_text'):
                    print(f"Warning: No audio segment path found or file does not exist for P:{participant_id} Turn Q:{qa_idx+1}. Original path: {audio_segment_path}")

            qa_pair['audio_features_xlsr53_path_relative'] = feature_vector_path_relative

            updated_qa_pairs.append(qa_pair)
        
        interview_data['qa_pairs'] = updated_qa_pairs
        updated_interviews_data.append(interview_data)

    # Save the updated JSON file with paths to the feature vectors
    updated_json_path = os.path.join(PROCESSED_DATA_DIR, PROCESSED_JSON_FILENAME.replace(".json", "_with_audio_features.json"))
    try:
        with open(updated_json_path, 'w') as f:
            json.dump(updated_interviews_data, f, indent=4)
        print(f"\nUpdated data with audio feature paths saved to {updated_json_path}")
    except Exception as e:
        print(f"Error saving updated JSON data: {e}")


if __name__ == '__main__':
    # This script assumes that `main_preprocess.py` has already been run
    # and `processed_daic_woz_interviews_qa.json` exists in PROCESSED_DATA_DIR
    # and the audio segments have been saved in PROCESSED_DATA_DIR/AUDIO_SEGMENTS_DIR_NAME
    
    input_json_file = os.path.join(PROCESSED_DATA_DIR, PROCESSED_JSON_FILENAME)
    
    if not os.path.exists(input_json_file):
        print(f"Error: Input JSON file not found at {input_json_file}")
        print("Please run the main_preprocess.py script first to generate audio segments and the initial JSON.")
    else:
        print("Starting audio feature extraction process...")
        process_all_audio_features(input_json_file)
        print("Audio feature extraction process complete.")

