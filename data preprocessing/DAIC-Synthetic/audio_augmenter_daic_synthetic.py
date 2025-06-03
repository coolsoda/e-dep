#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import os

try:
    from config_synthetic import FEATURE_FRAME_NUM_SWAPS
except ImportError:
    print("Warning: config_synthetic.py not found or FEATURE_FRAME_NUM_SWAPS not defined. Using default.")
    FEATURE_FRAME_NUM_SWAPS = 5


def random_feature_frame_swap(feature_sequence, num_swaps=FEATURE_FRAME_NUM_SWAPS):

    if not isinstance(feature_sequence, np.ndarray) or feature_sequence.ndim != 2:
        print(f"Warning: Expected 2D numpy array for feature_sequence, got {type(feature_sequence)}. Skipping swap.")
        return feature_sequence # Or handle error appropriately

    num_frames = feature_sequence.shape[0]
    if num_frames < 2:
        return feature_sequence 
        
    augmented_sequence = np.copy(feature_sequence)
    
    # Ensure num_swaps is not more than possible unique pairs to pick
    # or more than half the frames
    actual_swaps = min(num_swaps, num_frames // 2) 
    
    for _ in range(actual_swaps):
        idx1, idx2 = random.sample(range(num_frames), 2)
        
        # Swap the entire frame vectors
        frame_temp = np.copy(augmented_sequence[idx1, :])
        augmented_sequence[idx1, :] = augmented_sequence[idx2, :]
        augmented_sequence[idx2, :] = frame_temp
            
    return augmented_sequence

def augment_audio_features_for_fold_data(training_fold_data_list, 
                                        
                                         original_feature_sequence_dir): 
    
    print("Starting audio feature augmentation (feature-level frame swap)...")
    for interview_data in training_fold_data_list:
        p_id = interview_data['participant_id']
        for qa_pair in interview_data['qa_pairs']:

            turn_id_for_filename = qa_pair.get("turn_identifier_for_audio_aug", None)
            if turn_id_for_filename is None:
                # Fallback: if you have unique audio segment paths, derive from it
                if qa_pair.get('answer_audio_segment_path_absolute'):
                     turn_id_for_filename = os.path.basename(qa_pair['answer_audio_segment_path_absolute']).replace('.wav', '')
                else:
                    print(f"Warning: Cannot determine unique ID for audio segment in P_ID {p_id}. Skipping audio aug for this QA.")
                    qa_pair['audio_augmentation_type'] = "skipped_no_id"
                    continue
            
            original_feature_sequence_filename = f"{turn_id_for_filename}_xlsr53_sequence.npy"
            original_feature_sequence_path = os.path.join(original_feature_sequence_dir, f"{p_id}_P", original_feature_sequence_filename)


            if os.path.exists(original_feature_sequence_path):
                try:
                    original_sequence = np.load(original_feature_sequence_path) # Shape (num_frames, 1024)
                    augmented_sequence = random_feature_frame_swap(original_sequence)
                    pooled_augmented_feature = np.mean(augmented_sequence, axis=0) # Mean pooling
                    
        
                    qa_pair['audio_features_raw'] = pooled_augmented_feature.tolist() 
                    qa_pair['audio_augmentation_type'] = "feature_frame_swap"
                except Exception as e:
                    print(f"    Error augmenting audio features for {original_feature_sequence_path}: {e}")
                    qa_pair['audio_augmentation_type'] = "feature_frame_swap_error"
                    
                    if 'audio_features_raw' not in qa_pair and qa_pair.get('audio_features_xlsr53_path_relative'):
                        # Attempt to load original pooled feature if augmentation fails and it is not already there
                        try:
                            
                            from config_daic_woz import PROCESSED_DATA_DIR as DAIC_PROC_DIR
                            original_pooled_feature_path = os.path.join(DAIC_PROC_DIR, qa_pair['audio_features_xlsr53_path_relative'])
                            if os.path.exists(original_pooled_feature_path):
                                qa_pair['audio_features_raw'] = np.load(original_pooled_feature_path).tolist()
                        except Exception as load_e:
                             print(f"Could not load original pooled features for fallback: {load_e}")


            else:
                print(f"Warning: Original audio feature sequence not found at {original_feature_sequence_path} for P_ID {p_id}. Skipping audio aug.")
                qa_pair['audio_augmentation_type'] = "skipped_no_feature_seq"
                if 'audio_features_raw' not in qa_pair and qa_pair.get('audio_features_xlsr53_path_relative'):
                    try:
                        from config_daic_woz import PROCESSED_DATA_DIR as DAIC_PROC_DIR
                        original_pooled_feature_path = os.path.join(DAIC_PROC_DIR, qa_pair['audio_features_xlsr53_path_relative'])
                        if os.path.exists(original_pooled_feature_path):
                             qa_pair['audio_features_raw'] = np.load(original_pooled_feature_path).tolist()
                    except Exception as load_e:
                         print(f"Could not load original pooled features for fallback: {load_e}")

    print("Audio feature augmentation for fold complete.")
    return training_fold_data_list


if __name__ == '__main__':
    print("--- Testing Audio Augmenter (Feature Frame Swap) ---")
    
    dummy_original_feature_sequence_dir = "dummy_daic_audio_feature_sequences"
    dummy_pid = "test_300"
    dummy_participant_feature_seq_dir = os.path.join(dummy_original_feature_sequence_dir, f"{dummy_pid}_P")
    os.makedirs(dummy_participant_feature_seq_dir, exist_ok=True)

    dummy_seq_filename = f"{dummy_pid}_turn_1_answer_xlsr53_sequence.npy"
    dummy_feature_seq_path = os.path.join(dummy_participant_feature_seq_dir, dummy_seq_filename)
    
    # Create a dummy feature sequence
    dummy_sequence_data = np.random.rand(50, 1024)
    np.save(dummy_feature_seq_path, dummy_sequence_data)
    print(f"Created dummy feature sequence: {dummy_feature_seq_path}")

    dummy_fold_data_text_augmented = [
        {
            "participant_id": dummy_pid,
            "qa_pairs": [
                {
                    "question_text": "How have you been feeling lately?",
                    "answer_text": "I've been feeling pretty down and tired.",
                    "text_augmentation_type": "original",
                    "turn_identifier_for_audio_aug": f"{dummy_pid}_turn_1_answer" # For constructing path
                },
                
            ]
        }
    ]

    augmented_data_with_audio = augment_audio_features_for_fold_data(
        dummy_fold_data_text_augmented,
        dummy_original_feature_sequence_dir 
    )

    if augmented_data_with_audio:
        print("\nSample of data after audio feature augmentation:")
        for interview in augmented_data_with_audio:
            print(f"Interview ID: {interview['participant_id']}")
            for qa in interview['qa_pairs']:
                print(f"  Text Type: {qa['text_augmentation_type']}, Audio Aug Type: {qa.get('audio_augmentation_type')}")
                if 'audio_features_raw' in qa and qa['audio_features_raw'] is not None:
                    print(f"    Shape of pooled audio features: {np.array(qa['audio_features_raw']).shape}")
                else:
                    print("    No raw audio features found/generated for this QA.")
    else:
        print("No audio augmented data generated in test.")

    # Clean up dummy files and dirs
    if os.path.exists(dummy_feature_seq_path): os.remove(dummy_feature_seq_path)
    if os.path.exists(dummy_participant_feature_seq_dir) and not os.listdir(dummy_participant_feature_seq_dir): os.rmdir(dummy_participant_feature_seq_dir)
    if os.path.exists(dummy_original_feature_sequence_dir) and not os.listdir(dummy_original_feature_sequence_dir): os.rmdir(dummy_original_feature_sequence_dir)

    print("Audio augmentation (feature frame swap) test complete.")

