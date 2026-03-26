Dataset Preparation:
1. Prepare dataset, a text based sentences in a xlsx file format
2. Run clean dataset stage in yohanes.ipynb, where creates two columns: a clean version of sentences and deep_clean version
3. Run OpenAIEmbedding.ipynb if you want to embeddings the sentences using OpenAI embeddings model using API (need an OpenAI API key). This process will creates a JSON file with embeddings vectors on each sentences.
4. Run openai.ipynb to initiate identification of entities using OpenAI API gpt-5 model. This will create a ner column to the dataset that tags possible words into NER dictionary words.
5. 
