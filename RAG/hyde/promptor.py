CASE = """다음 법률 질문에 대한 가상의 판례를 참고한 법률 답변을 작성해줘
질문:{}
답변:"""

class Promptor:
    def build_prompt(self, query):
        return CASE.format(query)