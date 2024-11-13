
class Recomp:
    def __init__(self, generator, retriever):
        self.generator = generator
        self.retriever = retriever
    
    
    def compress_(self,query,retrieved_docs):
        for doc in retrieved_docs:
            context = doc.metadata["summary"]
            messages= self.get_message(query,context)
            completion = self.generator.chat.completions.create(model = "gpt-3.5-turbo",messages=messages)
            doc.metadata["summary"] = completion.choices[0].message.content
 
    def search(self,query,k=2):
        retrieved_docs = self.retriever.search(query,k)
        self.compress_(query,retrieved_docs)
        return retrieved_docs
    
    def get_message(self,query,context):
        messages=[
                {"role": "system", "content": "당신은 법률 판례를 요약하는 ai입니다. 질문에 답할 수 있도록 검색된 판례를 압축해주세요."},
                {
                    "role": "user",
                    "content": f"검색된 판례들을 주어진 질문에 답을 할 수 있도록 필수적인 정보만 남겨서 두 문장으로 압축해줘.\n 질문:{query} \n검색된 판례:{context}\n 압축된 판례: "
                }
            ]
        return messages