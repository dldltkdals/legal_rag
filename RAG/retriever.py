from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores.utils import DistanceStrategy
import os
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_core.documents import Document


    
def _load_documents():
    with open('documents.pickle', 'rb') as f:
       documents = pickle.load(f)
    new_doc = []
    for doc in documents:
        new_doc.append(Document(page_content = doc.page_content,
                            metadata = doc.metadata,
                            ))
    return new_doc

class BM25_Retriever:
    def __init__(self):
        self.retriever =  KiwiBM25Retriever.from_documents(_load_documents())
 
    def search(self,query,k = 2):
        self.retriever.k = k
        return self.retriever.invoke(query)
    def get_retriever(self):
        return self.retriever

class Retriever:
    def __init__(self,model_name = 'jhgan/ko-sroberta-nli'):        
        self.encoder =self._load_model(model_name)
        self.retriever = self._load_db()
 
    
    def get_retriever(self):
        return self.retriever
    
    def _load_model(self, model_name):
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True},
        )

    def _load_db(self):
        if os.path.exists("db"):
            return FAISS.load_local('./db/faiss',  embeddings= self.encoder,allow_dangerous_deserialization = True)
        else:
            db = FAISS.from_documents(
                                   documents = _load_documents(),
                                   embedding = self.encoder,
                                   distance_strategy = DistanceStrategy.COSINE,

                                  )
            db.save_local('./db/faiss')
            return db
    def search(self,query,k=2):
        documents = self.retriever.similarity_search(query,k=k)
        return documents
    





