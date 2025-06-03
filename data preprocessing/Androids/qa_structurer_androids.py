#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def structure_qa_from_transcribed_turns(turns_with_transcripts):
    """
    Structures transcribed turns into Question-Answer pairs.
    Each turn in turns_with_transcripts should have 'speaker', 'text', 
    'segmented_audio_path_relative'.
    """
    qa_pairs = []
    current_question_text = None
    current_question_audio_path = None

    for i in range(len(turns_with_transcripts)):
        turn = turns_with_transcripts[i]
        if turn['speaker'] == 'Interviewer':
            current_question_text = turn.get('text', "")
            current_question_audio_path = turn.get('segmented_audio_path_relative', None)
        
        elif turn['speaker'] == 'Participant' and current_question_text is not None:
            qa_pairs.append({
                'question_text': current_question_text,
                'question_audio_path_relative': current_question_audio_path,
                'answer_text': turn.get('text', ""),
                'answer_audio_path_relative': turn.get('segmented_audio_path_relative', None),
                'question_start_time': turns_with_transcripts[i-1].get('start_time') if i > 0 and turns_with_transcripts[i-1]['speaker'] == 'Interviewer' else None, 
                'question_stop_time': turns_with_transcripts[i-1].get('stop_time') if i > 0 and turns_with_transcripts[i-1]['speaker'] == 'Interviewer' else None,  
                'answer_start_time': turn.get('start_time'),
                'answer_stop_time': turn.get('stop_time'),
            })
            # Reset question, one Q-one A.
            # The current metadata_parser_androids creates alternating Interviewer/Participant turns.
            # So this one-to-one mapping fits this structure.
            current_question_text = None 
            current_question_audio_path = None

    return qa_pairs

