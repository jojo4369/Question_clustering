This code was built on Python 3.11.9 environtment.

This process adapted from A. Benayas, M. A. Sicilia, and M. Mora-Cantallops, “Automated Creation of an Intent Model for Conversational Agents,” Applied Artificial Intelligence, vol. 37, no. 1, p. 2164401, Dec. 2023, doi: 10.1080/08839514.2022.2164401. The main idea is text based dataset represented into 4 feature vectors: semantic vector (Vw), Lexical Vector/TF-IDF vector (Vk), Part of Speech (POS) tag vector (Vp), and LDA Probability topic vector (Vt). In addition, we add vector Named Entity Recognition (NER) (Vner) to represent the general representation of text in the form of entities word tags.

A. Dataset Preparation:
1. Prepare dataset, a text based sentences in a xlsx file format
2. Run stage-02 in yohanes.ipynb, where creates two columns: a clean version of sentences and deep_clean version
3. Run OpenAIEmbedding.ipynb if you want to embeddings the sentences using OpenAI embeddings (support multi languages) model using API (need an OpenAI API key). This process will creates a JSON file with embeddings vectors on each sentences (OpenAIEmbeddings_512_question.json, OpenAIEmbeddings_512_question_clean_simple.json).
4. Run openai.ipynb to initiate identification of entities using OpenAI API gpt-5 model. This will create a ner column to the dataset that tags possible words into pre-defined NER dictionary words.
5. The final output is an xlsx dataset (dataset.xlsx) where at least have original question, question_clean_simple, question_clean_deep, and ner column.
6. dataset.xlsx is real-world text-based user questions in the domain of Official Statistics (Statistics Indonesia - BPS, 2024) that has been anonimized and decomposed from original sentences.

B. Reproduce Clustering Process with HDBSCAN using Semantic vector (Vw), Lexical Vector/TF-IDF vector (Vk), Part of Speech (POS) tag vector (Vp), LDA Probability topic vector (Vt), and NER vector (additional) :
1. In yohanes.ipynb, using dataset.xlsx, run Stage-01 to load the dataset in the dataframe
2. Stage-03 to view the dataset, ensure requested colums are available (question, ner, question_clean_simple, question_clean_deep) and load to dataframe
3. 

