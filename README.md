This code was built on Python 3.11.9 environtment.

This process adapted from A. Benayas, M. A. Sicilia, and M. Mora-Cantallops, “Automated Creation of an Intent Model for Conversational Agents,” Applied Artificial Intelligence, vol. 37, no. 1, p. 2164401, Dec. 2023, doi: 10.1080/08839514.2022.2164401. 

The main idea is text based dataset represented into 4 feature vectors: semantic vector (Vw), Lexical Vector/Top-K n-gram vector (Vk), Part of Speech (POS) tag vector (Vp), and LDA Probability K-topic vector (Vt). 

In addition, we use HDBSCAN clustering algorithm instead of ITER-HDBSCAN and we add vector Named Entity Recognition (NER) (Vner) to represent the general representation of text in the form of entities word tags.

## A. Dataset Preparation:
1. Prepare dataset, a text based sentences in a xlsx file format
2. Run stage-02 in yohanes.ipynb, where creates two columns: a clean version of sentences and deep_clean version
3. Run OpenAIEmbedding.ipynb if you want to embeddings the sentences using OpenAI embeddings (support multi languages) model using API (need an OpenAI API key). This process will creates a JSON file with embeddings vectors on each sentences (OpenAIEmbeddings_512_question.json, OpenAIEmbeddings_512_question_clean_simple.json).
4. Run openai.ipynb to initiate identification of entities using OpenAI API gpt-5 model. This will create a ner column to the dataset that tags possible words into pre-defined NER dictionary words.
5. The final output is an xlsx dataset (dataset.xlsx) where at least have original question, question_clean_simple, question_clean_deep, and ner column.
6. dataset.xlsx is real-world text-based user questions in the domain of Official Statistics (Statistics Indonesia - BPS, 2024) that has been anonimized and decomposed from original sentences.

## B. Reproduce:
1. In yohanes.ipynb, using dataset.xlsx, run Stage-01 to load the dataset in the dataframe.
2. Stage-03 to view the dataset, ensure requested colums are available (question, ner, question_clean_simple, question_clean_deep) and load to dataframe.
3. Stage-07: create vector ner and stage-08, scale vector ner. At this stage, the process have **[vectors_ner_power]**.
4. Stage-09: create semantic vector (OpenAI embeddings) or use IndoSBERT embedding in stage-10 or use MiniLMEmbeddings in stage-11. At this stage, the process have **[vectors_vw]**.
5. since OpenAI embeddings (stage-09) is already normalized, skip stage-13. However, use stage-13 if you run with IndoSBERT (stage-10) or MiniLMEmbeddings (Stage-11).
6. Stage-14: build top-K ngram word(s) from column question_clean_deep (clean from stopwords). Use stage-15 if you want to load custom list Top-K ngram dictionary.
7. Stage-16: create lexical vector (Vk) and scale it (stage-17). At this stage, the process have **[vectors_vk_power]**.
8. Stage-18: build a syntactic vector (Vp) per row using Stanza-id Indonesia (use another language as per dataset used) and then run stage-19 to scale Vp. At this stage, the process have **[vectors_vp_power]**.
9. Stage-20: Calculate best K-topic using LDA model. Run this stage to get the coherence value per K value, select K value with higher coherence score. Using this K value, run stage-21 to build Vt/topic vector or just set K value in stage-21. Normalized Vt awith stage-22 and the process now have **[vectors_vt_normal]**.
10. Stage-23: concatenate **[vectors_vw]**, **[vectors_vk_power]**, **[vectors_vp_power]**, **[vectors_vt_normal]**, **[vectors_ner_power]**. at this stage the process have **[vectors_concat]**.
11. Stage-27: **[vectors_concat]** dimensionality reduction with UMAP. At this stage, the process have **[vectors_reduce]**.
12. Stage-28: Normalize **[vectors_reduce]** with L2, the output is **[vectors_norm]**. This **[vectors_norm]** will now be used as an input in HDBSCAN clustering algorithm.
13. Stage-29 and Stage-30: Data vectors visualization by reduce ist dimensionality to 3 with UMAP.
14. Stage-33: setting HDBSCAN parameter min_cluster_size and min_samples and run the algorithm.
15. Stage-37: Evaluate the cluster with silhouette_score, davies_bouldin_score, and calinski_harabasz_score. Those are internal evaluation metrics, where the assessment of the clustering quality is based solely on the dataset and the clustering results, and not on external, ground-truth labels.

## C. Experiment:
Using dataset.xlsx with 2756 records, we run HDBSCAN using several cases combination:
1. concat([vectors_vw], [vectors_vk_power], [vectors_vp_power], [vectors_vt_normal])
2. concat([vectors_vw], [vectors_vk_power], [vectors_vp_power], [vectors_vt_normal], [vectors_ner_power]) 
3. [vectors_vw]
4. [vectors_vk_power]
5. [vectors_vp_power]
6. [vectors_vt_normal]
7. [vectors_ner_power]

## D. Parameters:
1. For semantic vector **vw**, the embeddings use IndoSBERT model that support Indonesian sentences with vector length=256 (2756x256) and using OpenAI embeddings API (model: text-embedding-3-small) that support multi languages with length=512 (2756x512)
2. Using top-K = 30 with 1-4 ngram for lexical vector **vk** (2756x120)
3. For LDA topic vector **vt**, using T=8 (2756x8)
4. Part of speech (POS) vector **vp** using 17 predefined tags based on Universal POS Tag set (2756x17)
5. For vector entity **vner**, we define 28 entites based on SDMX concept on Data Structure Definition (measure, dimension, attribute, item list, etc.): ['ATTRIBUTE', 'DATA_VALUE', 'FACTORY', 'FILE_FORMAT', 'FREQUENCY', 'HOW', 'HOW_MANY', 'HOW_MUCH', 'INDICATOR', 'ITEM', 'LANGUAGE', 'LAW', 'NORP', 'NUMBER', 'ORG', 'OTHER_DIMENSION', 'PERSON', 'PRODUCT', 'QUESTION_MODAL', 'REF_AREA', 'TIME_PERIOD', 'UNIT_MEASURE', 'WHAT', 'WHEN', 'WHERE', 'WHICH', 'WHO', 'WHY'], therefore vector shape is (2756x28)
6. HDBSCAN min_cluster_size and min_samples range from 5, 10, 15, 25 (min_cluster_size=5 and min_samples=5, min_cluster_size=10 and min_samples=5, min_cluster_size=10 and min_samples=10, min_cluster_size=15 and min_samples=10, min_cluster_size=15 and min_samples=15, min_cluster_size=20 and min_samples=15, min_cluster_size=20 and min_samples=20, min_cluster_size=25 and min_samples=20, min_cluster_size=25 and min_samples=25)
7. Using UMAP dimensional reduction to 120 dimension (2756x120)

## E. Result
Using silhouette_score (higher is better) and davies_bouldin_score (lower is better), we evaluate best combination in features:
| no | Semantic Embeddings | Combination | min cluster size | min samples | Total Cluster | Noise | Without Noise | silhouette score | davies bouldin score |
|----|----------------------|------------------|------------------|-------------|---------------|-------|---------------|----------------|-------------------|
| 1  | OpenAI text-embedding-3-small | concat(vw, vk, vp, vt) | 10 | 10 | 82 | 187 | 2569 | <u>**0.9270**</u> | <u>**0.060**</u> |
| 2  | OpenAI text-embedding-3-small | concat(vw, vk, vp, vt, vner) | 10 | 10 | 82 | 303 | 2453 | 0.9210 | 0.084 |

| no | Semantic Embeddings | Combination | min cluster size | min samples | Total Cluster | Noise | Without Noise | silhouette score | davies bouldin score |
|----|----------------------|------------------|------------------|-------------|---------------|-------|---------------|----------------|-------------------|
| 1  | OpenAI text-embedding-3-small | concat(vw, vk, vp, vt) | 15 | 15 | 65 | 217 | 2539 | 0.9028 | 0.097 |
| 2  | OpenAI text-embedding-3-small | concat(vw, vk, vp, vt, vner) | 15 | 15 | 61 | 385 | 2371 | <u>**0.9234**</u> | <u>**0.069**</u> |

| no | Semantic Embeddings | Combination | min cluster size | min samples | Total Cluster | Noise | Without Noise | silhouette score | davies bouldin score |
|----|----------------------|------------------|------------------|-------------|---------------|-------|---------------|----------------|-------------------|
| 3  | OpenAI text-embedding-3-small | vw | 15 | 15 | 28 | 1136 | 1620 | 0.8184 | 0.156 |
| 3  | OpenAI text-embedding-3-small | vk | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3  | OpenAI text-embedding-3-small | vp | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3  | OpenAI text-embedding-3-small | vt | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3  | OpenAI text-embedding-3-small | vner | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## F. Conclusion






