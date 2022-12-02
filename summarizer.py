import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

# import os
import heapq
# import numpy as np
# import pandas as pd
# import sys
import streamlit as st
from io import StringIO

def preprocess(text):
    """
    INPUT:
    text - str. Original text.

    OUTPUT:
    clean_text - str. Tokenized and preprocessed text for word score computation.
    """
    clean_text = text.lower()
    clean_text = word_tokenize(clean_text)
    clean_text = [w for w in clean_text if w.isalnum()]

    return clean_text

def rank_sentence(text, clean_text):
    """
    Rank each sentence and return sentence score

    INPUT:
    text - str. Original text.
    clean_text - str. Tokenized and preprocessed text for word score computation.

    OUTPUT:
    sentence_score - dict. Sentence score
    top_n - int. Top n sentences to display.
    """
    sentences = sent_tokenize(text)
    stop_words = nltk.corpus.stopwords.words('english')

    word_count = {}
    for word in clean_text:
        if word not in stop_words:
            if word not in word_count.keys():
                word_count[word] = 1
            else:
                word_count[word] += 1

    sentence_score = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_count.keys():
                if len(sentence.split(' ')) < 30:
                    if sentence not in sentence_score.keys():
                        sentence_score[sentence] = word_count[word]
                    else:
                        sentence_score[sentence] += word_count[word]

    top_n = max(3, len(sentence_score)//4)

    return sentence_score, top_n

def generate_summary(text):
    """
    Generate summary

    INPUT:
    text - str. Original text.

    OUTPUT:
    summarized_text - str. Summarized text with each sentence on each line.
    """
    clean_text = preprocess(text)

    sentence_score, top_n = rank_sentence(text, clean_text)

    best_sentences = heapq.nlargest(top_n, sentence_score, key=sentence_score.get)

    summarized_text = []

    sentences = sent_tokenize(text)

    for sentence in sentences:
        if sentence in best_sentences:
            summarized_text.append(sentence)

    summarized_text = "\n".join(summarized_text)

    return summarized_text, len(sentences), len(best_sentences)

# generate summary
st.title("My First Text Summarizer 🍔")
st.subheader("Original Text")
text = ""
uploaded_file = st.file_uploader("Choose a text file or...")
if uploaded_file is not None:
    text = StringIO(uploaded_file.getvalue().decode("utf-8", errors='ignore')).read()
text = st.text_area(label="Paste your article here:", value=text)
st.subheader("Summary")
if text != "":
    summary, ori, suma = generate_summary(text)
    st.write(summary)
    st.write(f"Original text: {ori} sentences, {len(text.split())} words")
    st.write(f"Summary: {suma} sentences, {len(summary.split())} words")
