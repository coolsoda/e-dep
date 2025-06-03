#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import os
import numpy as np
import json
import time 


from config_androids import (
    PROCESSED_DATA_ANDROID_DIR,
    INPUT_JSON_FROM_AUDIO_EXTRACTION_ANDROID, 
    TEXT_FEATURES_ANDROID_DIR_NAME,
    PROCESSED_ANDROID_JSON_WITH_ALL_FEATURES_FILENAME, 
    XLM_R_MODEL_NAME,
    XLM_R_MAX_LENGTH
)


_XLM_R_MODEL_ANDROIDS = None
_XLM_R_TOKENIZER_ANDROIDS = None

def initialize_text_model_and_tokenizer_androids(model_name=XLM_R_MODEL_NAME):
    """Initializes and returns the XLM-R model and tokenizer for Androids."""
    global _XLM_R_MODEL_ANDROIDS, _XLM_R_TOKENIZER_ANDROIDS
    if _XLM_R_MODEL_ANDROIDS is None or _XLM_R_TOKENIZER_ANDROIDS is None:
        try:
            print(f"Initializing XLM-R model ('{model_name}') for Androids...")
            _XLM_R_TOKENIZER_ANDROIDS = XLMRobertaTokenizer.from_pretrained(model_name)
            _XLM_R_MODEL_ANDROIDS = XLMRobertaModel.from_pretrained(model_name)
            _XLM_R_MODEL_ANDROIDS.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _XLM_R_MODEL_ANDROIDS.to(device)
            print(f"Successfully loaded XLM-R model ('{model_name}') and tokenizer to {device} for Androids.")
        except Exception as e:
            print(f"Error loading XLM-R model/tokenizer ('{model_name}') for Androids: {e}")
            _XLM_R_MODEL_ANDROIDS = None
            _XLM_R_TOKENIZER_ANDROIDS = None
    return _XLM_R_MODEL_ANDROIDS, _XLM_R_TOKENIZER_ANDROIDS

def extract_text_features_androids(text_string, model, tokenizer, max_length=XLM_R_MAX_LENGTH):
    """
    Extracts features for a single text string using XLM-RoBERTa.
    Performs mean pooling of the last hidden states, considering the attention mask.
    """
    if not text_string or not text_string.strip() or text_string in ["ASR_TRANSCRIPTION_ERROR", "AUDIO_FILE_NOT_FOUND", "AUDIO_SEGMENTATION_FAILED"]:
        # For ASR might have failed or audio was missing
        return None 
        
    try:
        inputs = tokenizer(text_string, return_tensors="pt", max_length=max_length, 
                           padding="max_length", truncation=True)
        
        device = _XLM_R_MODEL_ANDROIDS.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        pooled_features = sum_embeddings / sum_mask
        
        return pooled_features.squeeze(0).cpu().numpy()

    except Exception as e:
        print(f"Error extracting text features for '{text_string[:50]}...': {e}")
        return None

def process_androids_text_features():
    """
    Iterates through Androids processed JSON data (after ASR and audio feature extraction),
    extracts text features for questions and answers, saves them, and updates the JSON.
    """
    xlmr_model, xlmr_tokenizer = initialize_text_model_and_tokenizer_androids()
    if xlmr_model is None or xlmr_tokenizer is None:
        print("Failed to initialize XLM-R model for Androids. Cannot extract text features.")
        return

    input_json_full_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, INPUT_JSON_FROM_AUDIO_EXTRACTION_ANDROID)
    try:
        with open(input_json_full_path, 'r', encoding='utf-8') as f:
            all_interviews_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file for Androids not found at {input_json_full_path}")
        print(f"Ensure '{INPUT_JSON_FROM_AUDIO_EXTRACTION_ANDROID}' exists in '{PROCESSED_DATA_ANDROID_DIR}'.")
        return
    except Exception as e:
        print(f"Error reading Androids JSON {input_json_full_path}: {e}")
        return
        
    dataset_text_features_output_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, TEXT_FEATURES_ANDROID_DIR_NAME)
    os.makedirs(dataset_text_features_output_path, exist_ok=True)
    print(f"Saving extracted Androids text features to: {dataset_text_features_output_path}")

    updated_interviews_data = []
    total_interviews = len(all_interviews_data)

    for interview_idx, interview_data in enumerate(all_interviews_data):
        participant_id = interview_data['participant_id']
        
        print(f"Extracting text features for Androids participant {participant_id} ({interview_idx+1}/{total_interviews})...")
        
        current_participant_text_feature_dir = os.path.join(dataset_text_features_output_path, participant_id)
        os.makedirs(current_participant_text_feature_dir, exist_ok=True)
        
        updated_qa_pairs = []
        for qa_idx, qa_pair in enumerate(interview_data['qa_pairs']):
            q_text = qa_pair.get('question_text') 
            a_text = qa_pair.get('answer_text')
            turn_identifier = qa_idx + 1
            
            # Question text (Interviewer ASR output)
            q_feature_vector_path_relative = None
            if q_text:
                q_feature_vector = extract_text_features_androids(q_text, xlmr_model, xlmr_tokenizer)
                if q_feature_vector is not None:
                    q_feature_filename = f"{participant_id}_qa_{turn_identifier}_question_xlmr.npy"
                    absolute_q_feature_path = os.path.join(current_participant_text_feature_dir, q_feature_filename)
                    np.save(absolute_q_feature_path, q_feature_vector)

                    q_feature_vector_path_relative = os.path.join(TEXT_FEATURES_ANDROID_DIR_NAME, participant_id, q_feature_filename)
            qa_pair['question_text_features_xlmr_path_relative'] = q_feature_vector_path_relative

            # Answer text (participant ASR)
            a_feature_vector_path_relative = None
            if a_text:
                a_feature_vector = extract_text_features_androids(a_text, xlmr_model, xlmr_tokenizer)
                if a_feature_vector is not None:
                    a_feature_filename = f"{participant_id}_qa_{turn_identifier}_answer_xlmr.npy"
                    absolute_a_feature_path = os.path.join(current_participant_text_feature_dir, a_feature_filename)
                    np.save(absolute_a_feature_path, a_feature_vector)
                    a_feature_vector_path_relative = os.path.join(TEXT_FEATURES_ANDROID_DIR_NAME, participant_id, a_feature_filename)
            qa_pair['answer_text_features_xlmr_path_relative'] = a_feature_vector_path_relative
            
            updated_qa_pairs.append(qa_pair)
        
        interview_data['qa_pairs'] = updated_qa_pairs
        updated_interviews_data.append(interview_data)


    final_output_json_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, PROCESSED_ANDROID_JSON_WITH_ALL_FEATURES_FILENAME)
    try:
        with open(final_output_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_interviews_data, f, indent=4, ensure_ascii=False)
        print(f"\nUpdated Androids data with all feature paths saved to {final_output_json_path}")
    except Exception as e:
        print(f"Error saving updated Androids JSON data: {e}")

# Main
if __name__ == '__main__':
    
    input_json_full_path = os.path.join(PROCESSED_DATA_ANDROID_DIR, INPUT_JSON_FROM_AUDIO_EXTRACTION_ANDROID)
    
    if not os.path.exists(input_json_full_path):
        print(f"Androids input JSON ('{INPUT_JSON_FROM_AUDIO_EXTRACTION_ANDROID}') not found in '{PROCESSED_DATA_ANDROID_DIR}'.")
        print("Please run the previous Androids preprocessing scripts first (ASR and participant audio feature extraction).")
    else:
        print(f"--- Starting Text Feature Extraction for Androids using {XLM_R_MODEL_NAME} ---")
        start_time = time.time()
        process_androids_text_features()
        end_time = time.time()
        print(f"--- Androids text feature extraction completed in {(end_time - start_time)/60:.2f} minutes ---")

