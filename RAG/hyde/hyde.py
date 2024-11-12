from generator import OpenAIGenerator
from promptor import Promptor

import os
import sys

from retriever import Retriever

from sentence_transformers import SentenceTransformer
import numpy as np


class HyDE:
    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher
    
    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents
    
    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = np.array(self.encoder.embed_query(c))
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        hyde_vector = np.mean(all_emb_c, axis=0)
      
        return hyde_vector
    def search(self, hyde_vector, k=10):
        hits = self.searcher.similarity_search_by_vector(hyde_vector, k=k)
        return hits
    

    def e2e_search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.search(hyde_vector, k=k)
        return hits
if __name__ == "__main__":
    retriever = Retriever()
    searcher = retriever.retriever
    encoder = retriever.model
    model_name="gpt-3.5-turbo"
    api_key = ""
    generator = OpenAIGenerator(model_name=model_name, api_key=api_key)
    promptor = Promptor()
    
    hyde = HyDE(promptor, generator,encoder,searcher)
    query = "폭행죄 처벌은 어떻게 되나요"
    hits = hyde.e2e_search(query, k=3)
    print(hits)