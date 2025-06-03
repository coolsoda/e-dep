#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import XLMRobertaModel, XLMRobertaTokenizer
import os
import numpy as np
import json
import time


from config import (
    PROCESSED_DATA_DIR,
    INPUT_JSON_FROM_AUDIO_EXTRACTION,
    TEXT_FEATURES_DAIC_WOZ_DIR_NAME,
    PROCESSED_DAIC_WOZ_JSON_WITH_ALL_FEATURES_FILENAME,
    XLM_R_MODEL_NAME,
    XLM_R_MAX_LENGTH
)

_XLM_R_MODEL = None
_XLM_R_TOKENIZER = None

def initialize_text_model_and_tokenizer(model_name=XLM_R_MODEL_NAME):
    """Initializes and returns the XLM-R model and tokenizer."""
    global _XLM_R_MODEL, _XLM_R_TOKENIZER
    if _XLM_R_MODEL is None or _XLM_R_TOKENIZER is None:
        try:
            print(f"Initializing XLM-R model ('{model_name}')...")
            _XLM_R_TOKENIZER = XLMRobertaTokenizer.from_pretrained(model_name)
            _XLM_R_MODEL = XLMRobertaModel.from_pretrained(model_name)
            _XLM_R_MODEL.eval()
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _XLM_R_MODEL.to(device)
            print(f"Successfully loaded XLM-R model ('{model_name}') and tokenizer to {device}.")
        except Exception as e:
            print(f"Error loading XLM-R model/tokenizer ('{model_name}'): {e}")
            _XLM_R_MODEL = None
            _XLM_R_TOKENIZER = None
    return _XLM_R_MODEL, _XLM_R_TOKENIZER

def extract_text_features(text_string, model, tokenizer, max_length=XLM_R_MAX_LENGTH):

    if not text_string or not text_string.strip():
        return None
        
    try:
        inputs = tokenizer(text_string, return_tensors="pt", max_length=max_length, 
                           padding="max_length", truncation=True)
        
        device = _XLM_R_MODEL.device
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

def process_daic_woz_text_features():
    xlmr_model, xlmr_tokenizer = initialize_text_model_and_tokenizer()
    if xlmr_model is None or xlmr_tokenizer is None:
        print("Failed to initialize XLM-R model. Cannot extract text features.")
        return

    input_json_full_path = os.path.join(PROCESSED_DATA_DIR, INPUT_JSON_FROM_AUDIO_EXTRACTION)
    try:
        with open(input_json_full_path, 'r', encoding='utf-8') as f:
            all_interviews_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file for DAIC-WOZ not found at {input_json_full_path}")
        print(f"Ensure '{INPUT_JSON_FROM_AUDIO_EXTRACTION}' exists in '{PROCESSED_DATA_DIR}'.")
        return
    except Exception as e:
        print(f"Error reading DAIC-WOZ JSON {input_json_full_path}: {e}")
        return
        
    # Base path for saving all DAIC-WOZ text features
    dataset_text_features_output_path = os.path.join(PROCESSED_DATA_DIR, TEXT_FEATURES_DAIC_WOZ_DIR_NAME)
    os.makedirs(dataset_text_features_output_path, exist_ok=True)
    print(f"Saving extracted DAIC-WOZ text features to: {dataset_text_features_output_path}")

    updated_interviews_data = []
    total_interviews = len(all_interviews_data)

    for interview_idx, interview_data in enumerate(all_interviews_data):
        participant_id = interview_data['participant_id']
        participant_feature_subdir_name = f"{participant_id}_P"
        
        print(f"Extracting text features for DAIC-WOZ participant {participant_id} ({interview_idx+1}/{total_interviews})...")
        
        current_participant_text_feature_dir = os.path.join(dataset_text_features_output_path, participant_feature_subdir_name)
        os.makedirs(current_participant_text_feature_dir, exist_ok=True)
        
        updated_qa_pairs = []
        for qa_idx, qa_pair in enumerate(interview_data['qa_pairs']):
            q_text = qa_pair.get('question_text')
            a_text = qa_pair.get('answer_text')
            turn_identifier = qa_idx + 1
            
            q_feature_vector_path_relative = None
            if q_text:
                q_feature_vector = extract_text_features(q_text, xlmr_model, xlmr_tokenizer)
                if q_feature_vector is not None:
                    q_feature_filename = f"{participant_id}_turn_{turn_identifier}_question_xlmr.npy"
                    absolute_q_feature_path = os.path.join(current_participant_text_feature_dir, q_feature_filename)
                    np.save(absolute_q_feature_path, q_feature_vector)
                    q_feature_vector_path_relative = os.path.join(TEXT_FEATURES_DAIC_WOZ_DIR_NAME, participant_feature_subdir_name, q_feature_filename)
            qa_pair['question_text_features_xlmr_path_relative'] = q_feature_vector_path_relative

            a_feature_vector_path_relative = None
            if a_text:
                a_feature_vector = extract_text_features(a_text, xlmr_model, xlmr_tokenizer)
                if a_feature_vector is not None:
                    a_feature_filename = f"{participant_id}_turn_{turn_identifier}_answer_xlmr.npy"
                    absolute_a_feature_path = os.path.join(current_participant_text_feature_dir, a_feature_filename)
                    np.save(absolute_a_feature_path, a_feature_vector)
                    a_feature_vector_path_relative = os.path.join(TEXT_FEATURES_DAIC_WOZ_DIR_NAME, participant_feature_subdir_name, a_feature_filename)
            qa_pair['answer_text_features_xlmr_path_relative'] = a_vector_path_relative
            
            updated_qa_pairs.append(qa_pair)
        
        interview_data['qa_pairs'] = updated_qa_pairs
        updated_interviews_data.append(interview_data)

    final_output_json_path = os.path.join(PROCESSED_DATA_DIR, PROCESSED_DAIC_WOZ_JSON_WITH_ALL_FEATURES_FILENAME)
    try:
        with open(final_output_json_path, 'w', encoding='utf-8') as f:
            json.dump(updated_interviews_data, f, indent=4, ensure_ascii=False)
        print(f"\nUpdated DAIC-WOZ data with all feature paths saved to {final_output_json_path}")
    except Exception as e:
        print(f"Error saving updated DAIC-WOZ JSON data: {e}")

# --- Main Execution ---
if __name__ == '__main__':
    
    input_json_full_path = os.path.join(PROCESSED_DATA_DIR, INPUT_JSON_FROM_AUDIO_EXTRACTION)
    
    if not os.path.exists(input_json_full_path):
        print(f"DAIC-WOZ input JSON ('{INPUT_JSON_FROM_AUDIO_EXTRACTION}') not found in '{PROCESSED_DATA_DIR}'.")
        print("Please run the previous DAIC-WOZ preprocessing scripts (including audio feature extraction if its output is used as input here) first.")
    else:
        print(f"--- Starting Text Feature Extraction for DAIC-WOZ using {XLM_R_MODEL_NAME} ---")
        start_time = time.time()
        process_daic_woz_text_features()
        end_time = time.time()
        print(f"--- DAIC-WOZ text feature extraction completed in {(end_time - start_time)/60:.2f} minutes ---")

