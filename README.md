# 법률 질의 응답 분야에서의 Advanced RAG 성능 비교 실험
본 연구는 검색결과의 질에 의해 좌우되는 RAG 모델의 한계를 극복하고, 법률 질의응답 분야에서 더욱 효과적인 챗봇 서비스를 개발하는 것을 목표로 합니다.
구체적으로, 본 연구는 다음과 같은 목표를 달성하고자 합니다.

**다양한 Advanced RAG 기법 비교 분석**: 기존 RAG 모델의 성능을 향상시키기 위해 제시된 다양한 Advanced RAG 기법들을 비교 분석하여 법률 분야에서의  각 기법의 특징과 효과를 파악합니다.

**실제 법률 질의응답 시나리오 적용**: 실제 법률 질의응답 시나리오에 다양한 Advanced RAG 기법을 적용하여 모델의 성능을 평가하고, 최적의 모델을 도출합니다.


# Advanced RAG
실험에 사용한 avanced RAG 기법은 아래와 같습니다.

**Re-Ranking**

검색 결과를 재순위화 해 중요한 문서를 상위에 위치 시키는 기법, 기존 RAG 모델은 bi-encoder 구조의 검색 과정에서 정보 손실이 발생하고, 검색 결과의 정확도가 떨어지는 문제가 있었습니다. Re-Ranking은 이러한 문제를 해결하기 위해 cross-encoder를 활용해 검색 결과를 재순위화하는 기술입니다.

<p align="center">
  <img width="720" alt="Re-Ranking pipeline" src="https://github.com/user-attachments/assets/e798a773-bfe7-472b-94b1-8f14162a3a9e">
</p>

**Recomp**

대규모 언어 모델을 이용해 검색된 문서에서 불필요한 정보를 제거하고 핵심 내용만을 추출하여 생성 모델에 전달하는 기술입니다. 이를 통해 중복되거나 관련 없는 정보를 제거하고, 생성 모델이 필요로 하는 핵심 정보만을 추출하여 압축된 정보 만으로 컨텍스트를 구성해 생성 과정의 효율성을 높입니다.
<p align="center">
  <img width="720" alt="Recomp pipeline" src="https://github.com/user-attachments/assets/4e907579-ceae-4e93-a971-d4c7f6e6c0f4">
</p>

**HyDE(Hypothetical Document Embeddings)**

HyDE는 대규모 언어 모델이 쿼리를 바탕으로 생성한 가상문서를 이용해 문서를 검색하는 방법입니다. 이를 통해 기존 쿼리를 정답 문서의 벡터공간에 더욱 가깝게 매핑할 수 있습니다.


<p align="center">
  <img width="720" alt="HyDE pipeline" src="https://github.com/user-attachments/assets/466c718d-1e97-4f91-8ce1-5db92cbdc933">
</p>

# 데이터 셋
### QA 데이터

LLM 학습 및 RAG 시스템 검증에 필요한 법률 QA 데이터셋을 구하기 위해  법률구조공단의 웹페이지 크롤링의 결과로 최종적으로 9,438개의 QA 데이터 셋을 확보 했습니다.해당 데이터는 노동, 주택임대, 상가임대,헌법, 민사, 형사 등 총 21개의 분야에 대한 법률 상담 사례 데이터로 구성되어있으며. 데이터 형식은 아래와 같습니다.
<p align="center">
  <img width="684" alt="qa data" src="https://github.com/user-attachments/assets/bdcd9bdd-a265-44b1-be1d-786b979dec4a">
</p>

### 판례 데이터 베이스

판례 데이터베이스를 구축하기 위해 ai hub에서 제공하는 법률/규정 분석 데이터를 사용했습니다. 해당 데이터는 60,000건 이상의 판례 데이터를 라벨링한 데이터로 판례요약, 판례에 대한 질문, 메타 정보가 포함되으며, 해당 데이터의 형식은 아래와 같습니다.
<p align="center">
  <img width="684" alt="case data" src="https://github.com/user-attachments/assets/2e474648-6d08-4ce2-85e6-1a73ec2f8c19">
</p>

# 실험 

### 실험 설정

- **Baseline:** 실험에서 비교군 설정을 위해 Bllossom 연구팀에서 llama-3기반의 모델을 대규모 한국어 데이터에 학습한 모델을 baseline으로 선정
- **Fine-tuning:** RAG과 fine-tuning 모델 간의 성능을 비교하기 위해 baseline 모델에 양자화, peft를 적용해 법률 도메인에 맞춰 법률 질의응답 데이터에 대해 fine-tuning을 진행
- **검증데이터:** 답변 생성 성능을 평가하기 위해 앞서 구축한 질의응답 데이터 중 일부를 답변생성 검증 데이터로 설정
- **개발환경:** 본 연구에서는 linux 환경에서 Tesla V100 GPU를 사용하여 구현 및 실험을 진행함

## 실험 결과

### 검색성능
<p align="center">
  <img width="693" alt="스크린샷 2024-11-13 오전 1 23 52" src="https://github.com/user-attachments/assets/a5b6951f-4a45-4de9-90c0-ae15300afdf0">
</p>
검색성능을 평가하기 위해 mAP(Mean Average Precision)를 사용하여 모델의 전체적인 검색 정확도를 측정했다. mAP는 정확도와 순위를 종합적으로 고려하는 지표로, 높은 값일수록 검색 결과의 질이 우수함을 의미한다.흥미롭게도, BM25 기반의 검색 방식이 Dense Retrieval 방식보다 월등히 높은 mAP 값을 기록했다. 이는 법률 데이터의 특성 상, 일반적인 언어 모델 기반의 Dense Retrieval 방식보다 전통적인 정보 검색 기법인 BM25가 더 적합하기 때문으로 판단된다. 법률 용어는 일반적인 단어에 비해 동의어가 적고 구체적인 의미를 지니는 경우가 많아, BM25와 같이 키워드 일치에 기반한 검색 방식이 법률 데이터의 특성을 더 잘 반영한 것으로 보인다.

###  검색된 문서의 효과

RAG 기법에 제공되는 검색된 문서의 개수가  답변 생성에 미치는 영향을 분석하기 위해  검색된 판례의 수를 증가시켜가며  bleu-1, bert score를 평가했다.
<p align="center">
  <img width="723" alt="스크린샷 2024-11-13 오전 1 26 34" src="https://github.com/user-attachments/assets/7ba9ce5a-a13b-4d26-9192-ed7ec1ee5f9d">
</p>
실험 결과 검색된 문서의 수가 증가할수록 모든 기법에서 BLEU-1 점수는 지속적으로 상승하는 경향을 보였다**.** 이는 BLEU-1 점수가 생성 문장과 참조 문장 간의 n-gram 일치율을 측정하는 지표이므로, 검색된 문서의 양이 증가할수록 생성 모델이 참조할 수 있는 단어 조합이 많아져 생성 문장의 다양성이 증가하기 때문이다.

BERT 점수는 re-ranking 기법이 적용된 모델의 경우 검색된 문서가 2개 이하일때  최고 성능을 보였으며, 그 이상의 문서가 제공되더라도 유의미한 성능 향상이 관찰되지 않았다.  이는 re-ranking이 검색된 문서를 재순위하면서 적은 수의 문서만으로 충분한 정보를 제공하기 때문으로 판단된다.

또한 RECOMP 기법은 검색된 문서의 수가 증가할수록 Bert score가 지속적으로 향상되는 경향을 보였다. 이는 RECOMP가 검색문서의 수가 증가해도 불필요한 정보를 효과적으로 제거하고 핵심 정보만을 압축해 생성모델에 전달함으로써, 생성된 답변과 실제 답변 간 의미적 유사성이 높아졌다  해석될 수 있다.

### 답변 예시
<p align="center">
  <img width="724" alt="image" src="https://github.com/user-attachments/assets/fd707dcc-3db7-4af5-8b14-655563fef460">
</p>
Advanced RAG 기법을 적용한 모델들은 복잡한 질문에 대해서도 명확하고 사실에 근거한 답변을 생성했다. 특히, HyDE 기법은 실제 법률 전문가가 제시한 근거와 유사한 근거를 제시하여 답변의 신뢰성을 높였습니다. 이는 Advanced RAG 기법이 복잡한 법률 논리 관계를 이해하고, 이를 바탕으로 논리적인 답변을 생성할 수 있음을 의미한다. 반면, RAG를 적용하지 않은 모델들은 복잡한 질문에 대해 어려움을 겪으며, 핵심 쟁점에서 벗어난 답변을 생성하거나, 잘못된 정보를 제공하는 경우가 많았다.


