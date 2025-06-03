#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import librosa
import soundfile as sf
import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from config_androids import ASR_MODEL_NAME, ASR_LANGUAGE

ASR_MODEL = None
ASR_PROCESSOR = None

def initialize_asr_model():
    global ASR_MODEL, ASR_PROCESSOR
    if ASR_MODEL is None or ASR_PROCESSOR is None:
        try:
            print(f"Loading ASR model: {ASR_MODEL_NAME}...")
            ASR_PROCESSOR = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
            ASR_MODEL = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ASR_MODEL.to(device)
            ASR_MODEL.config.forced_decoder_ids = None 
            if ASR_LANGUAGE: 
                 ASR_MODEL.config.forced_decoder_ids = ASR_PROCESSOR.get_decoder_prompt_ids(language=ASR_LANGUAGE, task="transcribe")

            print(f"ASR model {ASR_MODEL_NAME} loaded successfully to {device}.")
        except Exception as e:
            print(f"Error loading ASR model {ASR_MODEL_NAME}: {e}")
            ASR_MODEL = None 
            ASR_PROCESSOR = None
    return ASR_MODEL, ASR_PROCESSOR

def segment_and_save_turn_audio(full_audio_path, turn_info, turn_idx, participant_segment_dir):

    start_time_sec = turn_info['start_time']
    stop_time_sec = turn_info['stop_time']
    speaker = turn_info['speaker']
    
    if start_time_sec is None or stop_time_sec is None or stop_time_sec <= start_time_sec:
        print(f"Warning: Invalid timestamps for turn {turn_idx} ({speaker}): {start_time_sec}-{stop_time_sec}. Skipping segmentation.")
        return None

    segment_filename = f"{os.path.basename(full_audio_path).split('.')[0]}_turn_{turn_idx}_{speaker}.wav"
    output_segment_path = os.path.join(participant_segment_dir, segment_filename)
    
    try:
        y, sr_orig = librosa.load(full_audio_path, sr=None)
        
        # Whisper expects 16kHz
        if sr_orig != 16000:
            y = librosa.resample(y=y, orig_sr=sr_orig, target_sr=16000)
        sr_target = 16000

        start_sample = int(start_time_sec * sr_target)
        stop_sample = int(stop_time_sec * sr_target)
        
        start_sample = max(0, start_sample)
        stop_sample = min(len(y), stop_sample)

        if start_sample >= stop_sample:
            print(f"Warning: Calculated sample indices invalid for turn {turn_idx} ({speaker}) segment: {output_segment_path}. Skipping.")
            return None
            
        segment = y[start_sample:stop_sample]
        if len(segment) == 0:
            print(f"Warning: Extracted segment is empty for turn {turn_idx} ({speaker}): {output_segment_path}. Skipping.")
            return None

        os.makedirs(os.path.dirname(output_segment_path), exist_ok=True)
        sf.write(output_segment_path, segment, sr_target, subtype='PCM_16')
        return output_segment_path
        
    except Exception as e:
        print(f"Error segmenting turn {turn_idx} ({speaker}) for {full_audio_path} to {output_segment_path}: {e}")
        return None

def transcribe_audio_segment(audio_segment_path):
    global ASR_MODEL, ASR_PROCESSOR
    if ASR_MODEL is None or ASR_PROCESSOR is None:
        print("ASR model not initialized. Cannot transcribe.")
        return "ASR_MODEL_NOT_INITIALIZED" # Or raise an error

    if not audio_segment_path or not os.path.exists(audio_segment_path):
        print(f"Warning: Audio segment for ASR not found at {audio_segment_path}")
        return "AUDIO_FILE_NOT_FOUND"

    try:
        waveform, sr = librosa.load(audio_segment_path, sr=16000)

        input_features = ASR_PROCESSOR(waveform, sampling_rate=sr, return_tensors="pt").input_features
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_features = input_features.to(device)

        # Generate token ids
        predicted_ids = ASR_MODEL.generate(input_features, language=ASR_LANGUAGE if ASR_LANGUAGE else None)
        
        # Decode token ids to text
        transcription = ASR_PROCESSOR.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
        
    except Exception as e:
        print(f"Error transcribing audio segment {audio_segment_path}: {e}")
        return "ASR_TRANSCRIPTION_ERROR"

if __name__ == '__main__':
    # Example usage for testing this script
    from config_androids import PROCESSED_DATA_ANDROID_DIR, SEGMENTED_AUDIO_DIR_NAME, ANDROIDS_CORPUS_BASE_PATH
    
    print("Initializing ASR model for testing...")
    model, processor = initialize_asr_model()
    
    if model and processor:
        print("ASR Model Initialized.")
        sr_test = 16000
        duration_test = 10 # seconds
        dummy_audio_data = np.random.randn(sr_test * duration_test)
        
        dummy_pid_for_test = "test000"
        dummy_participant_audio_root = os.path.join(ANDROIDS_CORPUS_BASE_PATH, "HC")
        os.makedirs(dummy_participant_audio_root, exist_ok=True)
        dummy_full_audio_path_for_test = os.path.join(dummy_participant_audio_root, f"{dummy_pid_for_test}.wav")
        
        dummy_participant_segment_dir_for_test = os.path.join(PROCESSED_DATA_ANDROID_DIR, SEGMENTED_AUDIO_DIR_NAME, dummy_pid_for_test)

        try:
            sf.write(dummy_full_audio_path_for_test, dummy_audio_data, sr_test, subtype='PCM_16')
            print(f"Created dummy full audio: {dummy_full_audio_path_for_test}")

            dummy_turn_info = {'speaker': 'Participant', 'start_time': 2.0, 'stop_time': 5.0}
            
            saved_segment_path = segment_and_save_turn_audio(
                dummy_full_audio_path_for_test, 
                dummy_turn_info, 
                1, # turn_idx
                dummy_participant_segment_dir_for_test
            )

            if saved_segment_path:
                print(f"Dummy segment saved to: {saved_segment_path}")
                transcription = transcribe_audio_segment(saved_segment_path)
                print(f"Transcription for dummy segment: '{transcription}'")
                if os.path.exists(saved_segment_path): os.remove(saved_segment_path)
            else:
                print("Failed to create dummy segment for testing.")

        except Exception as e:
            print(f"Error in audio_tools_androids.py test: {e}")
        finally:
            if os.path.exists(dummy_full_audio_path_for_test):
                os.remove(dummy_full_audio_path_for_test)
                print(f"Removed dummy full audio: {dummy_full_audio_path_for_test}")
            if os.path.exists(dummy_participant_segment_dir_for_test) and not os.listdir(dummy_participant_segment_dir_for_test):
                 os.rmdir(dummy_participant_segment_dir_for_test)

    else:
        print("Failed to initialize ASR model for testing.")
    print("Audio tools test complete.")

