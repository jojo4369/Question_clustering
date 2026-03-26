from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class MiniLMEmbedding():
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