# import tensorflow_hub as hub

# class SentenceEmbedding(object):
#     """docstring for SentenceEmbedding"""

#     def __init__(self):
#         self.model = None
#         self.loadModel()

#     def loadModel(self):
#         if self.model is None:
#             print('Loading Tensorflow USE model....')
#             model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
#             self.model = hub.load(model_url)
#             print('Model Loaded.')

#     def embed(self, input_):
#         embedding = self.model(input_)
#         question_embeddings = embedding.numpy().tolist()
#         return question_embeddings

#     def getEmbeddings(self, data):
#         vector = []
#         start = 0
#         step = 10000

#         for i in range(int(len(data) / step)):
#             samples = data[start:start + step]
#             start += step
#             features = self.embed(samples)
#             vector += features

#         if len(data) % step != 0:
#             samples = data[start:]
#             features = self.embed(samples)
#             vector += features
#         return vector

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class SentenceEmbedding(object):
    def __init__(self):
        self.model = None
        self.loadModel()

    def loadModel(self):
        if self.model is None:
            print('Loading Huggingface sentence-transformers/all-MiniLM-L6-v2 model....')
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print('Model Loaded.')

    def encode(self, text):
        try:
            embeddings = self.model.encode(text)
            return embeddings
        except Exception as e:
            print(e)
            return
        

    def getEmbeddings(self, data):
        vector = []

        for text in tqdm(data):
            features = self.encode(text)
            vector.append(features)

        return vector