#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def extract_qa_pairs(structured_transcript):
    """
    Extracts Question-Answer pairs from a structured transcript.
    
    """
    qa_pairs = []
    current_question_info = None
    current_answer_segments = []

    for i, turn in enumerate(structured_transcript):
        speaker = turn['speaker']
        text = turn['value']
        start_time = turn['start_time']
        stop_time = turn['stop_time']

        if speaker == 'Ellie':
            if current_question_info and current_answer_segments: # Finalize previous QA
                answer_text_full = " ".join([seg['text'] for seg in current_answer_segments])
                ans_start = current_answer_segments[0]['start_time'] if current_answer_segments[0]['start_time'] is not None else current_question_info['stop_time']
                ans_stop = current_answer_segments[-1]['stop_time'] if current_answer_segments[-1]['stop_time'] is not None else (start_time if start_time is not None else current_question_info['stop_time'])
                
                qa_pairs.append({
                    'question_text': current_question_info['text'],
                    'question_start_time': current_question_info['start_time'],
                    'question_stop_time': current_question_info['stop_time'],
                    'answer_text': answer_text_full,
                    'answer_start_time': ans_start,
                    'answer_stop_time': ans_stop,
                })
            current_question_info = {'text': text, 'start_time': start_time, 'stop_time': stop_time}
            current_answer_segments = []
            
        elif speaker == 'Participant' and current_question_info:
            current_answer_segments.append({'text': text, 'start_time': start_time, 'stop_time': stop_time})
            
            if (i == len(structured_transcript) - 1) or                (i + 1 < len(structured_transcript) and structured_transcript[i+1]['speaker'] == 'Ellie'):
                if current_answer_segments: # Finalize current QA
                    answer_text_full = " ".join([seg['text'] for seg in current_answer_segments])
                    ans_start = current_answer_segments[0]['start_time'] if current_answer_segments[0]['start_time'] is not None else current_question_info['stop_time']
                    ans_stop = current_answer_segments[-1]['stop_time'] if current_answer_segments[-1]['stop_time'] is not None else                                (structured_transcript[i+1]['start_time'] if i + 1 < len(structured_transcript) and structured_transcript[i+1]['start_time'] is not None else                                 (stop_time if stop_time is not None else current_question_info['stop_time']))

                    qa_pairs.append({
                        'question_text': current_question_info['text'],
                        'question_start_time': current_question_info['start_time'],
                        'question_stop_time': current_question_info['stop_time'],
                        'answer_text': answer_text_full,
                        'answer_start_time': ans_start,
                        'answer_stop_time': ans_stop,
                    })
                    current_question_info = None 
                    current_answer_segments = []
    
    if current_question_info and current_answer_segments:
        answer_text_full = " ".join([seg['text'] for seg in current_answer_segments])
        ans_start = current_answer_segments[0]['start_time'] if current_answer_segments[0]['start_time'] is not None else current_question_info['stop_time']
        ans_stop = current_answer_segments[-1]['stop_time'] if current_answer_segments[-1]['stop_time'] is not None else current_question_info['stop_time']
        
        qa_pairs.append({
            'question_text': current_question_info['text'],
            'question_start_time': current_question_info['start_time'],
            'question_stop_time': current_question_info['stop_time'],
            'answer_text': answer_text_full,
            'answer_start_time': ans_start,
            'answer_stop_time': ans_stop,
        })
    elif current_question_info and not current_answer_segments:
        qa_pairs.append({
            'question_text': current_question_info['text'],
            'question_start_time': current_question_info['start_time'],
            'question_stop_time': current_question_info['stop_time'],
            'answer_text': "",
            'answer_start_time': current_question_info['stop_time'],
            'answer_stop_time': current_question_info['stop_time'],
        })
        
    return qa_pairs

# For simple testing 
# if __name__ == '__main__':

#     dummy_transcript_data = [
#         {'start_time': 0.0, 'stop_time': 1.0, 'speaker': 'Ellie', 'value': 'Hello, how are you?'},
#         {'start_time': 1.5, 'stop_time': 2.5, 'speaker': 'Participant', 'value': "I'm doing okay."},
#         {'start_time': 2.8, 'stop_time': 3.5, 'speaker': 'Participant', 'value': "Just a bit tired."},
#         {'start_time': 4.0, 'stop_time': 5.0, 'speaker': 'Ellie', 'value': 'What makes you tired?'},
#         {'start_time': 5.5, 'stop_time': 6.5, 'speaker': 'Participant', 'value': "Work mostly."},
#         {'start_time': 7.0, 'stop_time': 8.0, 'speaker': 'Ellie', 'value': 'I see.'} 
#     ]
#     qa_data = extract_qa_pairs(dummy_transcript_data)
#     print(f"Extracted {len(qa_data)} QA pairs from dummy data (first 2):")
#     for qa in qa_data[:3]: # Print all 3 for this dummy
#         print(f"  Q ({qa['question_start_time']}-{qa['question_stop_time']}): {qa['question_text']}")
#         print(f"  A ({qa['answer_start_time']}-{qa['answer_stop_time']}): {qa['answer_text']}\n")

