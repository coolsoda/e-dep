#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import json
from openai import OpenAI


try:
    from config_synthetic import (
        GPT_MODEL_NAME, 
        NUM_TEXT_ALTERNATIVES_PER_RESPONSE,
        LLM_TEMPERATURE,
        LLM_MAX_TOKENS_PER_REPHRASE,
        LLM_API_TIMEOUT_SECONDS,
        LLM_REQUEST_DELAY_SECONDS
    )
except ImportError:
    print("Warning: config_synthetic.py not found or not configured. Using default LLM settings.")
    GPT_MODEL_NAME = "gpt-4o"
    NUM_TEXT_ALTERNATIVES_PER_RESPONSE = 3
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS_PER_REPHRASE = 150
    LLM_API_TIMEOUT_SECONDS = 60
    LLM_REQUEST_DELAY_SECONDS = 1



_OPENAI_CLIENT = None

def get_openai_client():
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            _OPENAI_CLIENT = OpenAI(api_key=api_key, timeout=LLM_API_TIMEOUT_SECONDS)
            print(f"OpenAI client initialized with model {GPT_MODEL_NAME}.")
        else:
            print("OpenAI API key not found in environment variables. Text augmentation will use placeholders.")
    return _OPENAI_CLIENT

def generate_gpt_alternatives(original_participant_text, 
                              interviewer_question_text=None, 
                              num_alternatives=NUM_TEXT_ALTERNATIVES_PER_RESPONSE):

    client = get_openai_client()
    if not client:
        print(f"  LLM Placeholder: No API client. Returning original text for augmentation of '{original_participant_text[:30]}...'")
        return [f"Placeholder alternative {i+1} for: {original_participant_text}" for i in range(num_alternatives)]

    if not original_participant_text or not original_participant_text.strip():
        return [original_participant_text] * num_alternatives

    prompt_header = "Two speakers are having a conversation during an interview.\nThe interviewer is [Ellie], and the participant is [Participant].\n\nInput Conversation Context:\n"
    conversation_context = ""
    if interviewer_question_text:
        conversation_context += f"[Ellie] {interviewer_question_text}\n"
    conversation_context += f"[Participant] {original_participant_text}\n"
    
    instruction = f"\nPlease rephrase ONLY the [Participant]'s last statement shown above. Use different words and sentence structures while maintaining the core meaning and approximate sentiment. " \
                  f"Directly output the results without any explanations, preamble, or numbering like 'Alternative 1:'. Provide exactly {num_alternatives} distinct rephrased sentences, each on a new line."

    full_prompt = prompt_header + conversation_context + instruction
    
    alternatives = []
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL_NAME,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS_PER_REPHRASE * num_alternatives,
            n=1 
        )
        generated_text = response.choices[0].message.content
        raw_alternatives = [alt.strip() for alt in generated_text.split('\n') if alt.strip()]
        
        if len(raw_alternatives) >= num_alternatives:
            alternatives = raw_alternatives[:num_alternatives]
        else:
            alternatives = raw_alternatives
            alternatives.extend([original_participant_text] * (num_alternatives - len(alternatives)))

    except Exception as e:
        print(f"    Error calling {GPT_MODEL_NAME} for text '{original_participant_text[:60]}...': {e}")
        alternatives = [original_participant_text] * num_alternatives
    
    if LLM_REQUEST_DELAY_SECONDS > 0:
        time.sleep(LLM_REQUEST_DELAY_SECONDS)
    return alternatives

def augment_texts_for_fold_data(training_fold_data_list):

    if not get_openai_client(): 
        print("Text augmentation skipped due to OpenAI client initialization failure.")
    
    output_augmented_interviews = []
    for original_interview in training_fold_data_list:
        print(f"  Text augmenting for P_ID {original_interview['participant_id']}...")
        new_qa_list_for_this_interview = []
        for qa_pair in original_interview['qa_pairs']:
            original_participant_text = qa_pair.get('answer_text', "")
            interviewer_question_text = qa_pair.get('question_text', "")

            # 1. Add the original QA pair
            new_qa_list_for_this_interview.append({
                **qa_pair,
                "text_augmentation_type": "original"
            })

            if original_participant_text:
                alternatives = generate_gpt_alternatives(
                    original_participant_text, 
                    interviewer_question_text
                )
                for i, alt_text in enumerate(alternatives):
                    new_qa_list_for_this_interview.append({
                        **qa_pair, 
                        "answer_text": alt_text, 
                        "text_augmentation_type": f"{GPT_MODEL_NAME}_alt_{i+1}"
                    })
            else:
                 for i in range(NUM_TEXT_ALTERNATIVES_PER_RESPONSE):
                    new_qa_list_for_this_interview.append({
                        **qa_pair,
                        "answer_text": "", 
                        "text_augmentation_type": f"empty_orig_alt_{i+1}"
                    })
        
        output_augmented_interviews.append({
            **original_interview,
            "qa_pairs": new_qa_list_for_this_interview 
        })
    print(f"Text augmentation for fold complete. Processed {len(training_fold_data_list)} interviews.")
    return output_augmented_interviews

if __name__ == '__main__':
    print("--- Testing Text Augmenter ---")
    # dummy example
    dummy_fold_data = [
        {
            "participant_id": "test_300",
            "qa_pairs": [
                {
                    "question_text": "How have you been feeling lately?",
                    "answer_text": "I've been feeling pretty down and tired.",
                    "answer_audio_path_absolute": "path/to/audio1.wav" # Example field
                },
                {
                    "question_text": "What about your sleep?",
                    "answer_text": "It's been terrible, I can barely sleep.",
                    "answer_audio_path_absolute": "path/to/audio2.wav"
                }
            ]
        }
    ]
    
    print("Ensure your OPENAI_API_KEY environment variable is set for a real test.")
    augmented_data = augment_texts_for_fold_data(dummy_fold_data)
    
    if augmented_data:
        print("\nSample of augmented data:")
        for interview in augmented_data:
            print(f"Interview ID: {interview['participant_id']}")
            for qa in interview['qa_pairs']:
                print(f"  Type: {qa['text_augmentation_type']}, Answer: '{qa['answer_text'][:70]}...'")
    else:
        print("No augmented data generated in test.")
