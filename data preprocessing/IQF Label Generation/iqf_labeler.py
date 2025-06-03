#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import time
from openai import OpenAI

try:
    from config_llm_iqf import (
        IQF_LLM_MODEL_NAME,
        IQF_LLM_TEMPERATURE,
        IQF_LLM_MAX_TOKENS_RESPONSE,
        IQF_LLM_API_TIMEOUT_SECONDS,
        IQF_LLM_REQUEST_DELAY_SECONDS,
        IQF_CATEGORIES,
        IQF_DEFINITIONS_PROMPT
    )
except ImportError:
    print("FATAL: config_llm_iqf.py not found or not configured. Exiting.")
    exit()

# Assuming specific dataset configs are in their respective files. For example, to get input/output paths for DAIC-WOZ:
try:
    from config_daic_woz import (
        PROCESSED_DATA_DIR as DAIC_PROC_DIR,
        PROCESSED_DAIC_WOZ_JSON_WITH_ALL_FEATURES_FILENAME as DAIC_INPUT_JSON_NAME
    )
    # Define output name for DAIC-WOZ after IQF labeling
    DAIC_FINAL_JSON_WITH_IQF = DAIC_INPUT_JSON_NAME.replace(".json", "_with_iqf.json")
except ImportError:
    print("Warning: DAIC-WOZ config not fully found. Paths for DAIC-WOZ example might fail.")
    DAIC_PROC_DIR = None
    DAIC_INPUT_JSON_NAME = None
    DAIC_FINAL_JSON_WITH_IQF = None



_OPENAI_CLIENT_IQF = None

def get_openai_client_iqf():
    global _OPENAI_CLIENT_IQF
    if _OPENAI_CLIENT_IQF is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            _OPENAI_CLIENT_IQF = OpenAI(api_key=api_key, timeout=IQF_LLM_API_TIMEOUT_SECONDS)
            print(f"OpenAI client for IQF labeling initialized with model {IQF_LLM_MODEL_NAME}.")
        else:
            print("OpenAI API key not found in environment variables. IQF labeling will use placeholders.")
    return _OPENAI_CLIENT_IQF

def get_iqf_label_from_llm(current_interviewer_question, dialogue_history_turns=None):

    client = get_openai_client_iqf()
    if not client:
        print(f"  LLM Placeholder: No API client. Defaulting IQF for '{current_interviewer_question[:30]}...' to 'other'")
        return "other" 

    if not current_interviewer_question or not current_interviewer_question.strip():
        return "other" 

    history_str = "Previous conversation context (if any):\n"
    if dialogue_history_turns:
        history_str += "\n".join(dialogue_history_turns)
        history_str += "\n\n"
    else:
        history_str = "This is the beginning of the interview.\n\n"

    prompt_to_llm = (
        f"{history_str}"
        f"Interviewer's CURRENT QUESTION to classify: \"{current_interviewer_question}\"\n\n"
        f"{IQF_DEFINITIONS_PROMPT}\n"
        f"Output only the single best category label string for the CURRENT QUESTION from the list: {', '.join(IQF_CATEGORIES)}\n"
        f"Category:"
    )
    
    try:
        
        response = client.chat.completions.create(
            model=IQF_LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt_to_llm}],
            temperature=IQF_LLM_TEMPERATURE,
            max_tokens=IQF_LLM_MAX_TOKENS_RESPONSE 
        )
        predicted_label = response.choices[0].message.content.strip().lower() 
        
        valid_labels_lower = [cat.lower() for cat in IQF_CATEGORIES]
        if predicted_label not in valid_labels_lower:
            print(f"    Warning: LLM returned an unexpected label '{predicted_label}'. Defaulting to 'other'. Prompt was: {prompt_to_llm}")
            # Try to find closest match
            for valid_cat_lower in valid_labels_lower:
                if valid_cat_lower in predicted_label:
                    print(f"    Found substring match: assigning '{valid_cat_lower}'")
                    return valid_cat_lower 
            return "other"
        
        return IQF_CATEGORIES[valid_labels_lower.index(predicted_label)]

    except Exception as e:
        print(f"    Error calling {IQF_LLM_MODEL_NAME} for IQF of '{current_interviewer_question[:60]}...': {e}")
        return "other" # Fallback
    finally:
        if IQF_LLM_REQUEST_DELAY_SECONDS > 0:
            time.sleep(IQF_LLM_REQUEST_DELAY_SECONDS)


def label_iqfs_for_dataset(input_json_path, output_json_path, num_history_turns=2):
# Loads processed data, generates IQF labels for each interviewer question using an LLM,
# and saves the updated data.
    if not get_openai_client_iqf() and not os.environ.get("OPENAI_API_KEY"):
        print("IQF labeling will use placeholder 'other' for all questions due to missing API key.")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            all_interviews_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file for IQF labeling not found at {input_json_path}")
        return
    except Exception as e:
        print(f"Error reading JSON {input_json_path}: {e}")
        return
        
    print(f"Starting IQF labeling for {len(all_interviews_data)} interviews from {input_json_path}...")
    
    for interview_idx, interview_data in enumerate(all_interviews_data):
        participant_id = interview_data['participant_id']
        print(f"  Processing interview {interview_idx+1}/{len(all_interviews_data)} (ID: {participant_id}) for IQF labels...")
        
        turn_history_for_llm = []
        
        for qa_idx, qa_pair in enumerate(interview_data['qa_pairs']):
            current_question_text = qa_pair.get('question_text')
            current_answer_text = qa_pair.get('answer_text')


            relevant_history = []
            if num_history_turns > 0 and qa_idx > 0:
                start_hist_idx = max(0, qa_idx - num_history_turns)
                for prev_qa_idx in range(start_hist_idx, qa_idx):
                    prev_qa = interview_data['qa_pairs'][prev_qa_idx]
                    relevant_history.append(f"Interviewer: {prev_qa.get('question_text', '')}")
                    relevant_history.append(f"Participant: {prev_qa.get('answer_text', '')}")
            
            if current_question_text:
                iqf_label = get_iqf_label_from_llm(current_question_text, relevant_history)
                qa_pair['iqf_label'] = iqf_label
            else:
                qa_pair['iqf_label'] = "other" 

    try:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_interviews_data, f, indent=4, ensure_ascii=False)
        print(f"\nUpdated data with IQF labels saved to {output_json_path}")
    except Exception as e:
        print(f"Error saving updated JSON data with IQF labels: {e}")

if __name__ == '__main__':
  
    if DAIC_PROC_DIR and DAIC_INPUT_JSON_NAME and DAIC_FINAL_JSON_WITH_IQF:
        daic_input_json_for_iqf = os.path.join(DAIC_PROC_DIR, DAIC_INPUT_JSON_NAME)
        daic_output_json_with_iqf = os.path.join(DAIC_PROC_DIR, DAIC_FINAL_JSON_WITH_IQF)

        if os.path.exists(daic_input_json_for_iqf):
            print(f"\n--- Starting IQF Labeling for DAIC-WOZ ---")
            label_iqfs_for_dataset(daic_input_json_for_iqf, daic_output_json_with_iqf, num_history_turns=2)
        else:
            print(f"DAIC-WOZ input JSON for IQF labeling not found: {daic_input_json_for_iqf}")
            print("Ensure previous preprocessing steps for DAIC-WOZ are complete.")
    else:
        print("DAIC-WOZ paths not configured for IQF labeling example.")

    # Similar for EATD and Androids, usetheir respective config files for input/output paths.
   
    print("\nIQF Labeling script execution finished.")

