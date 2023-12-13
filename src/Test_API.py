#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:56:21 2023

@author: michiundslavki
"""

import requests

conversation_history_data = {
    "history": [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "Was ist die Rolle von Photovoltaik in der Energiewende?"}
    ]
    }
file_paths = {"filenames": [
    "/Users/michiundslavki/Dropbox/Deloitte/E_Control/RAG_bot_doc/docs/Photovoltaik.pdf", 
    
    "/Users/michiundslavki/Dropbox/Deloitte/E_Control/RAG_bot_doc/docs/Grüner_Wasserstoff.pdf"
    ]
    }

headers = {
    "accept": "application/json"
        }

url = "http://127.0.0.1:8000/chat"
url2 = "http://127.0.0.1:8000/paths"

response = requests.post(url, json=conversation_history_data)
print(response.content)
# file_name = requests.get(url2, json=file_paths)

# print(file_name.content)