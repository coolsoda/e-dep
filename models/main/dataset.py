#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import Dataset, DataLoader
import json
import os
import numpy as np

class InterviewDataset(Dataset):
    def __init__(self, manifest_file_path, processed_data_base_dir, 
                 tokenizer_for_padding=None,
                 max_seq_len=100): # Max number of QA pairs per interview
        
        super().__init__()
        print(f"Loading dataset manifest from: {manifest_file_path}")
        try:
            with open(manifest_file_path, 'r', encoding='utf-8') as f:
                self.interviews = json.load(f)
        except FileNotFoundError:
            print(f"FATAL: Manifest file not found at {manifest_file_path}")
            raise
        except Exception as e:
            print(f"FATAL: Error loading manifest file {manifest_file_path}: {e}")
            raise
            
        self.processed_data_base_dir = processed_data_base_dir
        self.max_seq_len = max_seq_len
        
        self.iqf_categories = [
            "Open Explore", "Elicit Change", "Info Request", "Structuring", 
            "Specific Probe", "Supportive", "Leading/Confirmatory", "Other"
        ]
        self.iqf_to_idx = {label: idx for idx, label in enumerate(self.iqf_categories)}
        self.num_iqf_classes = len(self.iqf_categories)

        print(f"Loaded {len(self.interviews)} interviews.")
        print(f"IQF categories used ({self.num_iqf_classes}): {self.iqf_categories}")

    def __len__(self):
        return len(self.interviews)

    def __getitem__(self, idx):
        interview_data = self.interviews[idx]
        participant_id = interview_data['participant_id']
        

        binary_label = torch.tensor(interview_data.get('binary_label', -1), dtype=torch.float) 
        regression_score = torch.tensor(interview_data.get('phq8_score', interview_data.get('sds_score', -1.0)), dtype=torch.float) 

        qa_pairs_data = interview_data['qa_pairs']
        
        # Initialize lists to hold features for all turns in this interview
        all_q_text_features = []
        all_a_text_features = []
        all_a_audio_raw_features = []
        all_iqf_labels = []
        turn_indices = []

        num_qa_pairs = len(qa_pairs_data)

        for i, qa_pair in enumerate(qa_pairs_data):
            if i >= self.max_seq_len: 
                num_qa_pairs = self.max_seq_len
                break

            # Question text features
            q_text_feat_path = qa_pair.get('question_text_features_xlmr_path_relative')
            if q_text_feat_path:
                try:
                    q_feat = np.load(os.path.join(self.processed_data_base_dir, q_text_feat_path))
                    all_q_text_features.append(torch.from_numpy(q_feat).float())
                except Exception as e:
                    
                    all_q_text_features.append(torch.zeros(768)) # Placeholder
            else:
                all_q_text_features.append(torch.zeros(768)) # Placeholder if path missing

            # Answer text features
            a_text_feat_path = qa_pair.get('answer_text_features_xlmr_path_relative')
            if a_text_feat_path:
                try:
                    a_feat = np.load(os.path.join(self.processed_data_base_dir, a_text_feat_path))
                    all_a_text_features.append(torch.from_numpy(a_feat).float())
                except Exception as e:
                    all_a_text_features.append(torch.zeros(768))
            else:
                all_a_text_features.append(torch.zeros(768))
            
            # Answer audio features
            
            a_audio_feat_path = qa_pair.get('participant_answer_audio_features_xlsr53_path_relative', 
                                          qa_pair.get('audio_features_xlsr53_path_relative'))
            if a_audio_feat_path:
                try:
                    audio_feat = np.load(os.path.join(self.processed_data_base_dir, a_audio_feat_path))
                    all_a_audio_raw_features.append(torch.from_numpy(audio_feat).float())
                except Exception as e:
            
                    all_a_audio_raw_features.append(torch.zeros(1024)) # Placeholder (1024 dim)
            else:
                all_a_audio_raw_features.append(torch.zeros(1024))

            # 4. IQF Labels (T_j)
            iqf_label_str = qa_pair.get('iqf_label', 'other') 
            iqf_idx = self.iqf_to_idx.get(iqf_label_str.lower(), self.iqf_to_idx['other']) 
            all_iqf_labels.append(torch.tensor(iqf_idx, dtype=torch.long))
            
            # 5. Turn indices for D-CoPE
            turn_indices.append(i)

        
        actual_seq_len = num_qa_pairs
        
        q_text_padding = torch.zeros(self.max_seq_len - actual_seq_len, 768)
        a_text_padding = torch.zeros(self.max_seq_len - actual_seq_len, 768)
        a_audio_padding = torch.zeros(self.max_seq_len - actual_seq_len, 1024)

        iqf_padding_value = self.iqf_to_idx['other']
        iqf_label_padding = torch.full((self.max_seq_len - actual_seq_len,), iqf_padding_value, dtype=torch.long)
        turn_indices_padding = torch.zeros(self.max_seq_len - actual_seq_len, dtype=torch.long)

        if actual_seq_len > 0:
            q_text_features_padded = torch.cat(all_q_text_features + [q_text_padding] if self.max_seq_len > actual_seq_len else all_q_text_features, dim=0) if all_q_text_features else q_text_padding
            a_text_features_padded = torch.cat(all_a_text_features + [a_text_padding] if self.max_seq_len > actual_seq_len else all_a_text_features, dim=0) if all_a_text_features else a_text_padding
            a_audio_features_padded = torch.cat(all_a_audio_raw_features + [a_audio_padding] if self.max_seq_len > actual_seq_len else all_a_audio_raw_features, dim=0) if all_a_audio_raw_features else a_audio_padding
            iqf_labels_padded = torch.cat(all_iqf_labels + [iqf_label_padding] if self.max_seq_len > actual_seq_len else all_iqf_labels, dim=0) if all_iqf_labels else iqf_label_padding
            turn_indices_padded = torch.cat([torch.tensor(turn_indices, dtype=torch.long)] + [turn_indices_padding] if self.max_seq_len > actual_seq_len else [torch.tensor(turn_indices, dtype=torch.long)], dim=0) if turn_indices else turn_indices_padding
        else: # case of empty qa_pairs list
            q_text_features_padded = q_text_padding
            a_text_features_padded = a_text_padding
            a_audio_features_padded = a_audio_padding
            iqf_labels_padded = iqf_label_padding
            turn_indices_padded = turn_indices_padding



        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
        attention_mask[:actual_seq_len] = True
        
        return {
            "participant_id": participant_id,
            "q_text_features": q_text_features_padded,    
            "a_text_features": a_text_features_padded,    
            "a_audio_raw_features": a_audio_features_padded, 
            "turn_indices": turn_indices_padded,       
            "attention_mask": attention_mask,          
            "iqf_labels": iqf_labels_padded,              
            "binary_label": binary_label,               
            "regression_score": regression_score,        
            "actual_seq_len": actual_seq_len           
        }

def collate_interviews(batch):
    """
    Collates a batch of interview data.
    Assumes __getitem__ already returns padded tensors of fixed seq_len.
    This function will stack these tensors.
    """

    if not batch:
        return {}

    p_ids = [item['participant_id'] for item in batch]
    q_text = torch.stack([item['q_text_features'] for item in batch])
    a_text = torch.stack([item['a_text_features'] for item in batch])
    a_audio_raw = torch.stack([item['a_audio_raw_features'] for item in batch])
    turn_indices = torch.stack([item['turn_indices'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    iqf_labels = torch.stack([item['iqf_labels'] for item in batch])
    
    binary_labels = torch.stack([item['binary_label'] for item in batch])
    regression_scores = torch.stack([item['regression_score'] for item in batch])
    actual_seq_lens = torch.tensor([item['actual_seq_len'] for item in batch], dtype=torch.long)

    return {
        "participant_ids": p_ids,
        "q_text_features": q_text,
        "a_text_features": a_text,
        "a_audio_raw_features": a_audio_raw,
        "turn_indices": turn_indices,
        "attention_mask": attention_mask,
        "iqf_labels": iqf_labels,
        "binary_labels": binary_labels,
        "regression_scores": regression_scores,
        "actual_seq_lens": actual_seq_lens
    }

