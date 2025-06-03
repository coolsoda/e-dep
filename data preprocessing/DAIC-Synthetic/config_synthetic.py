#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

GPT_MODEL_NAME = "gpt-4o"  # Or other models of choice"
NUM_TEXT_ALTERNATIVES_PER_RESPONSE = 3
LLM_TEMPERATURE = 0.7 
LLM_MAX_TOKENS_PER_REPHRASE = 100 # Estimated max tokens for one rephrased sentence
LLM_API_TIMEOUT_SECONDS = 60 # Timeout for API calls
LLM_REQUEST_DELAY_SECONDS = 1 


AUDIO_FRAME_LENGTH_MS_FOR_SWAP = 25 
NUM_AUDIO_SWAPS_PER_SEGMENT = 5    # pairs of frames to swap in an utterance

FEATURE_FRAME_NUM_SWAPS = 5


print("Synthetic data generation configurations loaded.")
if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable is not set. LLM-based text augmentation will fail or use placeholders.")

