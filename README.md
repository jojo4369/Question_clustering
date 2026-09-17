This code was built on Python 3.11.9 environtment.

## A. Dataset Preparation:
1. Prepare dataset, a text based sentences in a xlsx file format, run preprocessing stage using sources from preprocessing folder.
2. Run stage-02 in intent_mining_official_statistics.ipynb, where creates two columns: a clean version of sentences and deep_clean version
3. Run OpenAIEmbedding.ipynb if you want to embeddings the sentences using OpenAI embeddings (support multi languages) model using API (need an OpenAI API key). This process will creates a JSON file with embeddings vectors on each sentences (OpenAIEmbeddings_512_question.json, OpenAIEmbeddings_512_question_clean_simple.json).
4. The final output is an xlsx dataset (dataset_intent_mining_framework.xlsx) where at least have original question, question_clean_simple, and question_clean_deep column.

## B. Reproduce:
1. In intent_mining_official_statistics.ipynb, using dataset.xlsx, run Stage-01 to load the dataset in the dataframe.
2. Stage-03 to view the dataset, ensure requested colums are available (question, ner, question_clean_simple, question_clean_deep) and load to dataframe.
3. Stage-04: create semantic vector (OpenAI embeddings) or use IndoSBERT embedding in stage-05. At this stage, the process have **[vectors_vw]**.
5. since OpenAI embeddings (stage-04) is already normalized, skip stage-06. However, use stage-06 if you run with IndoSBERT (stage-05).
6. Stage-07: build top-K ngram word(s) from column question_clean_deep (clean from stopwords). Use stage-08 if you want to load custom list Top-K ngram dictionary.
7. Stage-09: create lexical vector (Vk) and scale it (stage-10). At this stage, the process have **[vectors_vk_power]**.
8. Stage-11: build a syntactic vector (Vp) per row using Stanza-id Indonesia (use another language as per dataset used) and then run stage-12 to scale Vp. At this stage, the process have **[vectors_vp_power]**.
9. Stage-13: Run this stage to get the coherence value per K value, select K value with optimal coherence score. Using this K value, run stage-14 to build Vt/topic vector or just set K value in stage-14. Normalized Vt with stage-15 and the process now have **[vectors_vt_normal]**.
10. Stage-16: concatenate **[vectors_vw]**/**[vectors_vw_normal]**, **[vectors_vk_power]**, **[vectors_vp_power]**, **[vectors_vt_normal]**. At this stage the process have **[vectors_concat]**.
11. Stage-17: **[vectors_concat]** dimensionality reduction with UMAP. At this stage, the process have **[vectors_reduce]**.
12. Stage-18: Normalize **[vectors_reduce]** with L2, the output is **[vectors_norm]**. This **[vectors_norm]** will now be used as an input in HDBSCAN clustering algorithm.
13. Stage-19 and Stage-20: Data vectors visualization by reduce ist dimensionality to 3 with UMAP.
14. Stage-21: setting HDBSCAN parameter min_cluster_size and min_samples and run the algorithm.
15. Stage-22: Evaluate the cluster with silhouette_score, davies_bouldin_score, and calinski_harabasz_score. Those are internal evaluation metrics, where the assessment of the clustering quality is based solely on the dataset and the clustering results, and not on external, ground-truth labels.
16. Stage-25: Intent Labeling stage with top-20 terms on each cluster with TF-IDF.
17. Stage-26: Calculate each pair of cluster silimarity.


