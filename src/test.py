#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 17:06:11 2023

@author: michiundslavki
"""
import os
import fitz
from collections import defaultdict

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import openai
from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
import docx2txt
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

# load_dotenv("/Users/michiundslavki/Dropbox/Deloitte/E_Control/chat-with-documents-langchain/Microservice_Chat_bot/.env")
# # Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")

# def generate_story(scenario):
#     template = f"""
#     generate a small concise story with about 30 words and in German
#     CONTENT: {scenario}
#     STORY:
#     """
#     #prompt = PromptTemplate(template=template, input_variables=["scenario"])
#     chat = ChatOpenAI(temperature=0, openai_api_key=openai.api_key)
#     messages = [
#     SystemMessage(
#         content="You are a helpful assistant that tells stories about image captions."
#     ),
#     HumanMessage(
#         content=template
#     ),
#     ]
#     story = chat(messages)
#     return(story.content) 

# print(generate_story("three workers in the building side"))

from pydantic import BaseModel, Field
from typing import List

#load_dotenv("/Users/michiundslavki/Dropbox/Deloitte/E_Control/Archive/chat-with-documents-langchain/Microservice_Chat_bot/.env")

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")

class FilePaths(BaseModel):
    filenames: List[str] = Field(
        ...,  # Use '...' for required fields
        description="List of file paths",
        example=["/path/to/file1.txt", "/path/to/file2.txt"]
    )

# Example usage
file_paths = FilePaths(filenames=["/path/to/file1.txt", "/path/to/file2.txt"])
print(type(file_paths.filenames))
"""
HERE comes a object oriented version of the File Proccessor

"""
from abc import ABC, abstractmethod

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
# HERE
        
    
uploaded_files =[ "/Users/michiundslavki/Downloads/Schlussfolgerung_corr_Michael.docx",
"/Users/michiundslavki/Downloads/requirements.txt",
"/Users/michiundslavki/Dropbox/Deloitte/E_Control/Natural_Language/chat_bot_doc/docs/Grüner_Wasserstoff.pdf"]


file_path_clusterer = FilePathsModifier()


clustered_file_paths = file_path_clusterer.cluster_files_by_extension(uploaded_files)

all_text = " "

for file_type, file_path in clustered_file_paths.items():
        processor = ProcessorFactory.get_processor(file_type)
        if processor:
            for file in file_path:
                text = processor.process(file)
                all_text += text
        else:
            print(f"Unsupported file type: {file}")

# # Textsplitter
# chunk_size=5000
# chunk_overlap=1000
# text_splitter = CharacterTextSplitter(
#                  separator="\n",
#                  chunk_size=chunk_size, #1000
#                  chunk_overlap=chunk_overlap,
#                  length_function=len
#                 )
# # Embedings for the Vector Db, here the same as of the LLM
# embeddings = OpenAIEmbeddings()

# # Create a Vectordatabase
# db_creator = DataBaseCreator(all_text, text_splitter, embeddings)

# #insider the communicator class

# db_creator.split_text()
# db_creator.create_vectorDB()



# #Testing
# user_input= "Was ist Photovoltaik"
# docs = db_creator.similarity_search(user_input)

# # the LLM back bone of the 
# llm = OpenAI()
# chain = load_qa_chain(llm, chain_type="refine") # stuff, map_reduce


# print(docs)

# we want to create an Object that takes a string as input and creates a db and then with get method gets 

# Usag

# all_text = ""
# for uploaded_file in uploaded_files:
#     processor = ProcessorFactory.get_processor(uploaded_file)
#     if processor:
#         text = processor.process(uploaded_file)
#         all_text += text
#     else:
#         print(f"Unsupported file type: {uploaded_file}")
    



# txt_processor = TextProcessor()
# doc_processor = DocxProcessor()
# all_text = doc_processor.process(file_path)
# #txt_processor.process(file_path)#pdf_processor.process(file_path)

# print(all_text[:100])

def extract_images_from_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Iterate through each page
    for i in range(len(doc)):
        # Get the page
        page = doc[i]

        # List of images in the page
        image_list = page.get_images(full=True)
        print(image_list)

        # Extract each image
        for image_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Save the image
            image_filename = f"image{page.number+1}_{image_index+1}.png"
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            print(f"Saved {image_filename}")

# Path to your PDF file
pdf_path = "/Users/michiundslavki/Dropbox/Deloitte/E_Control/Natural_Language/chat_bot_doc/docs/Grüner_Wasserstoff.pdf"
extract_images_from_pdf(pdf_path)

    