from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain.retrievers import ContextualCompressionRetriever

class Reranker:
    def __init__(self,retriever,reranker,top_k=3):
        self.retriever = retriever.get_retriever().as_retriever(search_kwargs = {'k':20})
        self.reranker = reranker
        self.top_k = top_k
        self.rerank_retriever = self._load_reranker()
        
    def _load_reranker(self):
        compressor =  CrossEncoderReranker(model=self.reranker,top_n = self.top_k)
        return ContextualCompressionRetriever(base_compressor= compressor, base_retriever=self.retriever)
    
    def search(self, query):
        return self.rerank_retriever.invoke(query)