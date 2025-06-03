#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import json
import time
import re
from openai import OpenAI

# Make sure config_synthetic.py and config.py are in the same directory or accessible
from config import PROCESSED_DATA_DIR as DAIC_PROCESSED_DIR 
from config import PROCESSED_JSON_FILENAME as DAIC_JSON_FILENAME
from config_synthetic import SYNTHETIC_TEXT_AUGMENTED_DIR, SYNTHETIC_TEXT_AUGMENTED_FILENAME, GPT_MODEL_NAME


try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable not found. API calls may fail.")
        # client = OpenAI(api_key="sk-xxxxxxx") # Replace with your key
except Exception as e:
    print(f"Error initializing OpenAI client: {e}. API calls will likely fail.")
    client = None

def generate_gpt4o_alternatives(original_participant_text, interviewer_question_text=None, num_alternatives=3):

    if not client:
        print("OpenAI client not initialized. Returning original text.")
        return [original_participant_text] * num_alternatives

    if not original_participant_text or not original_participant_text.strip():
        return [original_participant_text] * num_alternatives

    prompt_header = "Two speakers are having a conversation during an interview.\nThe interviewer is [Ellie], and the participant is [Participant].\n\nInput Conversation:\n"
    conversation_context = ""
    if interviewer_question_text:
        conversation_context += f"[Ellie] {interviewer_question_text}\n"
    conversation_context += f"[Participant] {original_participant_text}\n"
    
    instruction = f"\nPlease rephrase the last statement made by [Participant] above using different words while maintaining the same meaning and approximate sentiment. "                   f"Directly output the results without any explanations or numbering. Provide {num_alternatives} distinct rephrased sentences, each on a new line."

    full_prompt = prompt_header + conversation_context + instruction
    
    alternatives = []
    try:
        print(f"  Querying GPT-4o for: '{original_participant_text[:60]}...'")
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.7,
            max_tokens=100 * num_alternatives,
            n=1
        )
        generated_text = response.choices[0].message.content
        # Assuming each rephrased sentence is on a new line
        raw_alternatives = [alt.strip() for alt in generated_text.split('\n') if alt.strip()]
        
        if len(raw_alternatives) >= num_alternatives:
            alternatives = raw_alternatives[:num_alternatives]
        else:
            alternatives = raw_alternatives
            alternatives.extend([original_participant_text] * (num_alternatives - len(alternatives)))
        
        print(f"    ...Got {len(alternatives)} alternatives.")

    except Exception as e:
        print(f"Error calling GPT-4o for text '{original_participant_text[:60]}...': {e}")
        alternatives = [original_participant_text] * num_alternatives # Fallback
    
    time.sleep(1)
    return alternatives


def augment_training_texts_from_daic(daic_processed_json_path, training_ids_file=None):
    try:
        with open(daic_processed_json_path, 'r', encoding='utf-8') as f:
            all_daic_interviews = json.load(f)
    except FileNotFoundError:
        print(f"Error: DAIC-WOZ processed JSON file not found at {daic_processed_json_path}")
        return []

    training_pids = None
    if training_ids_file and os.path.exists(training_ids_file):
        with open(training_ids_file, 'r') as f:
            training_pids = {line.strip() for line in f if line.strip()}
        print(f"Loaded {len(training_pids)} training participant IDs.")
    else:
        print("Warning: No training_ids_file provided or found. Augmenting all interviews in the input JSON.")

    synthetic_interviews_output = []
    
    for interview_idx, original_interview in enumerate(all_daic_interviews):
        participant_id = original_interview['participant_id']
        
        if training_pids and participant_id not in training_pids:
            continue

        print(f"Augmenting text for interview {interview_idx+1}/{len(all_daic_interviews)} (ID: {participant_id})")
        
        augmented_qa_list_for_this_interview = []
        for qa_pair_idx, original_qa_pair in enumerate(original_interview['qa_pairs']):
            original_participant_text = original_qa_pair.get('answer_text', "")
            interviewer_question_text = original_qa_pair.get('question_text', "")

            augmented_qa_list_for_this_interview.append({
                **original_qa_pair,
                "augmentation_type": "original_text"
            })
            
            alternative_texts = generate_gpt4o_alternatives(
                original_participant_text, 
                interviewer_question_text, 
                num_alternatives=3
            )
            
            for i, alt_text in enumerate(alternative_texts):
                augmented_qa_list_for_this_interview.append({
                    **original_qa_pair,
                    "answer_text": alt_text, 
                    "augmentation_type": f"text_alt_{i+1}"
                })
        
        synthetic_interview_entry = {
            **original_interview, 
            "qa_pairs": augmented_qa_list_for_this_interview, 
            "is_text_augmented": True
        }
        synthetic_interviews_output.append(synthetic_interview_entry)

    print(f"Text augmentation complete. Output contains {len(synthetic_interviews_output)} interviews (training set augmented).")
    return synthetic_interviews_output


if __name__ == '__main__':
    # Path to the JSON output from DAIC-WOZ preprocessing 
    input_daic_json = os.path.join(DAIC_PROCESSED_DIR, DAIC_JSON_FILENAME)
    

    training_ids_filepath = os.path.join(DAIC_PROCESSED_DIR, "training_pids.txt")

    if not os.path.exists(input_daic_json):
        print(f"Input JSON {input_daic_json} for DAIC-WOZ not found. Please run DAIC-WOZ preprocessing first.")
    else:
        print(f"Starting text augmentation using {GPT_MODEL_NAME}...")
        synthetic_data = augment_training_texts_from_daic(input_daic_json, training_ids_filepath)
        
        if synthetic_data:
            output_path = os.path.join(SYNTHETIC_TEXT_AUGMENTED_DIR, SYNTHETIC_TEXT_AUGMENTED_FILENAME)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(synthetic_data, f, indent=4, ensure_ascii=False)
            print(f"Text-augmented synthetic data description saved to {output_path}")
            if synthetic_data:
                print(f"Sample of first augmented interview (ID: {synthetic_data[0]['participant_id']}):")
                # Print first few QA pairs to show original + alternatives
                for qa_sample in synthetic_data[0]['qa_pairs'][:4]: # show one original + 3 alternatives
                    print(f"  Type: {qa_sample['augmentation_type']}, Answer: '{qa_sample['answer_text'][:80]}...'")
        else:
            print("No synthetic data generated.")

