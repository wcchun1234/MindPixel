#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 13:27:19 2023
@author: wcchun
"""

import PyPDF2
import os
import re

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ' '.join([page.extract_text() for page in reader.pages])
    return text

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove dates in the format of dd/mm/yyyy
    text = re.sub(r'\b\d{1,2}/\d{1,2}/\d{4}\b', '', text)

    # Remove non-ASCII characters (removes non-English text and special characters)
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove standalone numbers and single characters
    text = re.sub(r'\b\w{1,1}\b', '', text)
    text = re.sub(r'\b\d+\b', '', text)

    # Remove specific technical jargon, names, and course references (customize as needed)
    text = re.sub(r'email@example\.com|professor john doe|course code abc123', '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()  # Strip leading and trailing whitespace

def save_text_to_file(text, output_file):
    # Appending text to the same file
    with open(output_file, 'a', encoding='utf-8') as file:
        file.write(text + '\n')

if __name__ == '__main__':
    pdf_directory = '/Users/wcchun/cityu/SM3750 Machine Learning for Artists/assignment_2/lecture notes sample/'
    output_file = '/Users/wcchun/cityu/SM3750 Machine Learning for Artists/assignment_2/outcome/cumulative_text_database.txt'

    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            
            try:
                print(f"Processing file: {filename}")
                extracted_text = extract_text_from_pdf(pdf_path)
                cleaned_text = clean_text(extracted_text)
                save_text_to_file(cleaned_text, output_file)
                print(f"Appended cleaned text from {filename} to database")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
