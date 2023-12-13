#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:32:02 2023

@author: michiundslavki
"""

# %%writefile backend.py
import os
from collections import defaultdict
from typing import Literal
from dotenv import load_dotenv
import openai
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from abc import ABC, abstractmethod
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import docx2txt
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
load_dotenv("/Users/michiundslavki/Dropbox/Deloitte/E_Control/Archive/chat-with-documents-langchain/Microservice_Chat_bot/.env")


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
class FilePaths(BaseModel):
    filenames: list[str] = Field(
        description="List of file paths",
        example=["/path/to/file1.txt", "/path/to/file2.txt"]
    )

class FilePathsModifier():
    def cluster_files_by_extension(self, directory_path):
        file_clusters = defaultdict(list)
        
        # Iterate over all files in the specified directory
        for file in directory_path:
            # Check if it's a file and not a directory
            #if os.path.isfile(os.path.join(directory_path, file)):
                # Extract the file extension
            _, file_extension = os.path.splitext(file)
                # Add the file to the corresponding list in the dictionary
            file_clusters[file_extension].append(file)
    
        return file_clusters


# Abstract base class
class FileProcessor(ABC):
    @abstractmethod
    def process(self, uploaded_file):
        pass
class TxtReader():
    def __init__(self, file_path):
        self.file_path = file_path
        
    
    def read_text_file(self,):
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return file.read()


class PdfProcessor(FileProcessor):
    def process(self, uploaded_file):
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    
# Text file processor
class TextProcessor(FileProcessor):
    def process(self, uploaded_file):
        txt_reader = TxtReader(uploaded_file)
        return txt_reader.read_text_file()

# DOCX file processor
class DocxProcessor(FileProcessor):
    def process(self, uploaded_file):
        return docx2txt.process(uploaded_file)

   
# File Processor Factory
class ProcessorFactory:
    @staticmethod
    def get_processor(file_type):
        if file_type == ".pdf": #application/pdf":
            return PdfProcessor()
        elif file_type == ".txt" :#text/plain":
            return TextProcessor()
        elif file_type == ".docx":#"application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return DocxProcessor()
        else:
           return None
class BotBackend(): 
    
    def __init__(self,):
        self._filenames = [
            "/Users/michiundslavki/Dropbox/Deloitte/E_Control/Natural_Language/chat_bot_doc/docs/Photovoltaik.pdf", 
            
            "//Users/michiundslavki/Dropbox/Deloitte/E_Control/Natural_Language/chat_bot_doc/docs/Grüner_Wasserstoff.pdf"
            ]
        #self._response = "I'm an LLM chat bot"
        self.history = [{"role": "user", "content": "Hello, how are you?"},{"role": "assistant", "content": "I'm fine, thank you!"}]

    def get_last_user_input(self,):
        full_question = False
        last_dict = self.history[-1]
        # Check if 'role' key exists and its value is 'user'
        if last_dict.get('role') == 'user':
           last_question = last_dict['content']
           full_question = True
        else:
            last_question = "Ich habe die Frage nicht verstanden bitte stellen sie Sie nochmals!"
        
        return last_question, full_question
            
        
    def add_chain_response(self, response):
        add_dict = {"role": "assistant", "content": response}
        self.history.append(add_dict)
        return None
    
    @property
    def filenames(self):
        """The getter method."""
        return self._filenames  
    
    @filenames.setter
    def filenames(self, new_filnames):
        """The setter method."""
        if new_filnames != self._filenames:
            self._filenames = new_filnames
        return None
    
    # @property
    # def response(self):
    #     """The getter method."""
    #     return self._response 
    
    # @response.setter
    # def response(self, new_response):
    #     """The setter method."""
    #     self._response = new_response
    #     return None

class DataBaseCreator():
    def __init__(self, string_text, text_splitter, embeddings):
        self.all_text = string_text
        self.text_splitter = text_splitter
        self.embeddings = embeddings

    def split_text(self,):
        self.chunks = self.text_splitter.split_text(self.all_text)
        return self.chunks
    # create embeddings
    def create_vectorDB(self,):
        self.knowledge_base = FAISS.from_texts(texts=self.chunks, embedding=self.embeddings)  
        return None
    def similarity_search(self,user_input):
        docs = self.knowledge_base.similarity_search(user_input)
        return docs
    

file_path_clusterer = FilePathsModifier()
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="refine") # stuff, map_reduce
backend_bot = BotBackend()

@app.get("/")
async def health_check():
    return {"status": "OK!"}


@app.post("/chat")
async def llm_response(history: ConversationHistory) -> dict:
    # Step 0: Receive the API payload as a dictionary
    history = history.dict()
    backend_bot.history = history["history"]
    
    #clustering the files in different types
    clustered_file_paths = file_path_clusterer.cluster_files_by_extension(backend_bot.filenames)
    #convert all into one string
    all_text = " "

    for file_type, file_path in clustered_file_paths.items():
            processor = ProcessorFactory.get_processor(file_type)
            if processor:
                for file in file_path:
                    text = processor.process(file)
                    all_text += text
            else:
                print(f"Unsupported file type: {file}")

    # Textsplitter
    chunk_size=5000
    chunk_overlap=1000
    text_splitter = CharacterTextSplitter(
                     separator="\n",
                     chunk_size=chunk_size, #1000
                     chunk_overlap=chunk_overlap,
                     length_function=len
                    )
    # Embedings for the Vector Db, here the same as of the LLM
    embeddings = OpenAIEmbeddings()
    
    # Create a Vectordatabase
    db_creator = DataBaseCreator(all_text, text_splitter, embeddings)
    #insider the communicator class
    db_creator.split_text()
    db_creator.create_vectorDB()

    last_question, success = backend_bot.get_last_user_input()
    
    if success:
        docs = db_creator.similarity_search(last_question)
        response_chain = chain.run(input_documents=docs, question=last_question)
        backend_bot.add_chain_response(response_chain)
    else:
        backend_bot.add_chain_response(last_question)

    print(last_question, backend_bot.history)
    return {
            'history': backend_bot.history
            } 
# {
#         "message": llm_response.choices[0]["message"],
#         "token_usage": llm_response["usage"],
#     }




@app.get("/paths")
async def file_paths(filenames: FilePaths)-> dict:
    
    filenames  = filenames.dict()
   
    backend_bot.filenames = filenames['filenames'] # this is a list at this level
    
    return {
        "filenames": backend_bot.filenames,
        "status": "ok"
    }
    
    


