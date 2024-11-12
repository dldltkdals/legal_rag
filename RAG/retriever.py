from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings
import pickle
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import os
from langchain_teddynote.retrievers import KiwiBM25Retriever
from langchain_core.documents import Document

def result_format(documents):
        result = []
        for doc in documents:
            result.append({
                "question":doc.page_content, 
                "case_title": doc.metadata["case_title"],
                "context":doc.metadata["summary"],
                })
        return result
    
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
    def set_k(self,k):
        self.retriever.k = k
    def search(self,query):
        return result_format(self.retriever.invoke(query))

class Retriever:
    def __init__(self,model_name = 'jhgan/ko-sroberta-nli'):        
        self.model =self._load_model(model_name)
        self.retriever = self._load_db()
        self.top_k = 3
       
    def _load_model(self, model_name):
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device':'cuda'},
            encode_kwargs={'normalize_embeddings':True},
        )
    def set_k(self,k):
        self.top_k = k
    def _load_db(self):
        if os.path.exists("db"):
            return FAISS.load_local('./db/faiss',  embeddings= self.model,allow_dangerous_deserialization = True)
        else:
            db = FAISS.from_documents(
                                   documents = _load_documents(),
                                   embedding = self.model,
                                   distance_strategy = DistanceStrategy.COSINE,

                                  )
            db.save_local('./db/faiss')
            return db
    def search(self,query):
        documents = self.retriever.similarity_search(query,k=self.top_k)
        return result_format(documents)
    
class Reranker_Retriever(Retriever):
    def __init__(self,k=2):
        super().__init__()
        self.reranker = HuggingFaceCrossEncoder(model_name="Dongjin-kr/ko-reranker")
        self.set_k(k)
        self.compression_retriever = self._load_reranker(self.top_k)
        
    def _load_reranker(self,top_k):
        rerank_model =  HuggingFaceCrossEncoder(model_name="Dongjin-kr/ko-reranker")
        compressor = CrossEncoderReranker(model=rerank_model,top_n = top_k)
        return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=self.retriever.as_retriever(search_kwargs = {'k':20}))
    def search(self, query):
        return result_format(self.compression_retriever.invoke(query))



if __name__ == "__main__":
    retrieval = Retriever()
    
 
    print(retrieval.search("6개월 동안 근무를 했는데 퇴직금을 받을 수 있나요?"))
    print("="*30)
    print(retrieval.search("6개월 동안 근무를 했는데 퇴직금을 받을 수 있나요?"))
