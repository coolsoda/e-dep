#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

def parse_eatd_qa_file(file_path):

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) >= 2:
            question = lines[0].strip()
            answer = lines[1].strip()
            return {'question': question, 'answer': answer}
        else:
            print(f"Warning: File {file_path} has fewer than 2 lines. Expected Question and Answer.")
            if len(lines) == 1:
                 return {'question': lines[0].strip(), 'answer': ""}
            return {'question': "", 'answer': ""} 
            
    except FileNotFoundError:
        print(f"Error: Transcript file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error processing transcript file {file_path}: {e}")
        return None

if __name__ == '__main__':
    # Example usage for testing this script
    from config_eatd import EATD_BASE_PATH
    
    example_participant_folder = os.path.join(EATD_BASE_PATH, 't_1')
    example_txt_file = os.path.join(example_participant_folder, 'negative.txt')

    if os.path.exists(example_txt_file):
        qa_content = parse_eatd_qa_file(example_txt_file)
        if qa_content:
            print(f"Content from {example_txt_file}:")
            print(f"  Question: {qa_content['question']}")
            print(f"  Answer: {qa_content['answer']}")
        else:
            print(f"Failed to parse {example_txt_file}.")
    else:
        print(f"Example .txt file not found: {example_txt_file}")
        print("Please ensure EATD_BASE_PATH in config_eatd.py is correct and participant folders/files exist.")

