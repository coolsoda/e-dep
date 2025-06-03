#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

IQF_LLM_MODEL_NAME = "gpt-4o"  
IQF_LLM_TEMPERATURE = 0.2     
IQF_LLM_MAX_TOKENS_RESPONSE = 50
IQF_LLM_API_TIMEOUT_SECONDS = 60
IQF_LLM_REQUEST_DELAY_SECONDS = 1


IQF_CATEGORIES = [
    "open-ended",
    "change talk",
    "neutral information gathering",
    "transitional",
    "specific probing",
    "supportive",
    "other"
]


IQF_DEFINITIONS_PROMPT = """
Here are the definitions for the Interviewer Question Functions (IQFs):
1.  open-ended: Questions that encourage participants to express themselves freely and broadly about a topic, including their experiences, thoughts, or feelings, typically inviting more than just short or specific answers.
2.  change talk: Questions directed at helping the participant express their motivations, reasons, desires, abilities, or needs related to making behavioral, cognitive, or situational changes.
3.  neutral information gathering: Questions seeking specific factual details, clarification of objective information, or direct answers to concrete inquiries, often expecting a constrained or brief response.
4.  transitional: Utterances (questions or statements) used by the interviewer to organize the conversation, manage transitions between topics, introduce or close segments, summarize points, or manage interview logistics.
5.  specific probing: Follow-up questions designed to elicit more detailed information, elaboration, or specific examples regarding a topic or statement previously introduced by the participant, generally in a neutral, non-leading manner.
6.  supportive: Statements or questions that primarily convey empathy, understanding, validation of the participant's feelings or experiences, offer encouragement, build rapport, or affirm their strengths.
7.  other: Utterances that do not clearly fit into any of the other defined functional categories. This can include very short backchannels, incomplete or interrupted sentences, unintelligible speech, or off-topic remarks not related to structuring.

Please classify the interviewer's CURRENT QUESTION into one of these exact categories.
"""

if not os.environ.get("OPENAI_API_KEY"):
    print("WARNING: OPENAI_API_KEY environment variable is not set. LLM-based IQF labeling will fail or use placeholders.")

