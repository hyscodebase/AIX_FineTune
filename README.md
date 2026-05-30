# BERTweet / RoBERTa H100 Domain Bias Fine-tuning Notebooks

이 폴더에는 서로 다른 도메인 컨셉의 H100용 실험 노트북 10개가 들어 있습니다.
각 노트북은 기존 B0 baseline → domain-specific treatment matrix → 평가 → 편향 예측/추천 타임라인을 유지합니다.

## 공통 목적
파인튜닝 방법(Full FT, LoRA, Adapter), 파라미터(LoRA rank, Adapter bottleneck), 도메인 특성에 따라
성능과 편향 정도가 어떻게 달라지는지 기록하고, 원하는 bias_score에 가까운 설정을 추천하는 노하우를 축적합니다.

## 노트북 목록
- `01_social_twitter_sentiment.ipynb`: 소셜미디어 감성 도메인 (Social Media Sentiment)
- `02_finance_news_sentiment.ipynb`: 금융 뉴스 감성 도메인 (Finance News Sentiment)
- `03_movie_review_sentiment.ipynb`: 영화 리뷰 감성 도메인 (Movie Review Sentiment)
- `04_product_review_sentiment.ipynb`: 제품 리뷰 감성 도메인 (Product Review Sentiment)
- `05_social_emotion_classification.ipynb`: 소셜 감정 분류 도메인 (Social Emotion Classification)
- `06_hate_speech_detection.ipynb`: 혐오표현 탐지 도메인 (Hate Speech Detection)
- `07_offensive_language_detection.ipynb`: 공격적 표현 탐지 도메인 (Offensive Language Detection)
- `08_irony_detection.ipynb`: 아이러니 탐지 도메인 (Irony Detection)
- `09_news_topic_classification.ipynb`: 뉴스 토픽 분류 도메인 (News Topic Classification)
- `10_question_intent_classification.ipynb`: 질문 의도 분류 도메인 (Question Intent Classification)

## 기본 실행법
1. Colab 또는 Jupyter H100 런타임에서 노트북을 엽니다.
2. 첫 번째 환경 셀을 실행합니다.
3. 필요하면 런타임을 재시작합니다.
4. 설정 셀에서 `RUN_PROFILE`, `BASE_MODEL_IDS`, `LORA_RANKS`, `ADAPTER_BOTTLENECKS`, `TARGET_BIAS_SCORE`를 조정합니다.
5. 메인 실험 셀을 실행합니다.

## 결과 파일
각 노트북은 `/content/<notebook_id>_outputs/` 아래에 다음 파일을 생성합니다.

- `tables/final_ranking.csv`
- `tables/bias_summary.csv`
- `tables/bias_group_detail.csv`
- `tables/target_bias_recommendations.csv`
- `tables/bias_modeling_dataset.csv`
- `report.md`
- `<notebook_id>_outputs.zip`

## Bias score 의미
`bias_score`는 다음 지표를 조합한 종합 편향 관찰 점수입니다.
낮을수록 현재 평가셋에서 관찰된 편향이 낮다는 의미지만, 편향이 없음을 보장하지는 않습니다.

- 집단별 F1 gap
- 집단별 Accuracy gap
- 집단별 prediction-rate gap
- Counterfactual flip rate
- 클래스별 집단 F1 gap


## Fix note

This fixed package replaces JSON-style `null` literals in Python code cells with Python `None`, preventing `NameError: name 'null' is not defined`.


## GPU autodetect patch

This version no longer stops only because the device name does not contain `H100`. It allows H100, A100, RTX PRO 6000 Blackwell, Blackwell-named CUDA GPUs, or CUDA GPUs with at least 40GB VRAM, and automatically chooses batch sizes from detected VRAM.
