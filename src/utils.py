#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:32:02 2023

@author: michiundslavki
"""

# %%writefile backend.py
import os
from typing import Literal

import openai
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = "You are a comic book assistant. You reply to the user's question strictly from the perspective of a comic book assistant. If the question is not related to comic books, you politely decline to answer."


class Conversation(BaseModel):
    role: Literal["assistant", "user"]
    content: str


class ConversationHistory(BaseModel):
    history: list[Conversation] = Field(
        example=[
            {"role": "user", "content": "tell me a quote from DC comics about life"},
        ]
    )


@app.get("/")
async def health_check():
    return {"status": "OK!"}


@app.post("/chat")
async def llm_response(history: ConversationHistory) -> dict:
    # Step 0: Receive the API payload as a dictionary
    history = history.dict()

    # Step 1: Initialize messages with a system prompt and conversation history
    messages = [{"role": "system", "content": system_prompt}, *history["history"]]

    # Step 2: Generate a response
    llm_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=messages
    )

    # Step 3: Return the generated response and the token usage
    return {
        "message": llm_response.choices[0]["message"],
        "token_usage": llm_response["usage"],
    }