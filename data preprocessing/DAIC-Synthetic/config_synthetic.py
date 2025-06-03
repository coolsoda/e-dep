#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

SYNTHETIC_TEXT_AUGMENTED_DIR = "F:/DAIC-WOZ/processed_data_synthetic/" 
SYNTHETIC_TEXT_AUGMENTED_FILENAME = "daic_synthetic_text_augmented.json"

# LLM Configuration
GPT_MODEL_NAME = "gpt-4o"

if not os.path.exists(SYNTHETIC_TEXT_AUGMENTED_DIR):
    try:
        os.makedirs(SYNTHETIC_TEXT_AUGMENTED_DIR)
    except Exception as e:
        print(f"Error creating directory {SYNTHETIC_TEXT_AUGMENTED_DIR}: {e}")

