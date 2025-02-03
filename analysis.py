import torch
from transformers import PreTrainedTokenizerFast
from transformers.models.bart import BartForConditionalGeneration
import csv
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# 모델 바이너리 파일 경로
model_binary_path = r"D:\pt\KoBART-summarization-main\kobart_summary"

# KoBART 모델 및 토크나이저 로드
tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v1')
model = BartForConditionalGeneration.from_pretrained(model_binary_path)

# 감정 분석 파이프라인 로드
sentiment_analyzer = pipeline("sentiment-analysis")

# 감정 및 긍정/부정 빈도수 저장을 위한 딕셔너리
emotion_count = defaultdict(int)
sentiment_count = defaultdict(int)

# CSV 파일 읽기
with open('cons.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        # 입력 텍스트 추출 (각 줄을 감정 분석하고 요약 생성)
        satoori_input_text = row[0]  # 첫 번째 컬럼에 텍스트가 있다고 가정
        
        # 텍스트를 KoBART로 요약
        input_ids = tokenizer.encode(satoori_input_text, return_tensors="pt", max_length=1800, truncation=True)
        output_ids = model.generate(input_ids, max_length=1800, min_length=0, length_penalty=2.0, num_beams=4, early_stopping=True)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # 감정 분석
        sentiment_result = sentiment_analyzer(output_text)
        
        # 감정 및 긍정/부정 빈도수 업데이트
        emotion = output_text.split()[-1]  # 요약된 텍스트의 마지막 단어를 감정으로 간주
        emotion_count[emotion] += 1
        
        sentiment = sentiment_result[0]['label']
        sentiment_count[sentiment] += 1
        
        # 결과 출력
        print(f"Original Text: {satoori_input_text}")
        print(f"Summarized Text: {output_text}")
        print(f"Sentiment: {sentiment} (confidence: {sentiment_result[0]['score']:.4f})")
        print("--------------------------------------------------")

# 감정 빈도수 그래프 그리기
emotions = list(emotion_count.keys())
counts = list(emotion_count.values())

# 색상 설정 (colormap 사용)
colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))  # viridis colormap 사용

plt.figure(figsize=(10, 5))
bars = plt.bar(emotions, counts, color=colors)
plt.title('Emotion Frequency')
plt.xlabel('Emotion')
plt.ylabel('Frequency')

# 각 막대에 색상과 빈도수 표시
for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{count}', ha='center', va='bottom')

plt.savefig('emotion_frequency.png')  # 그래프를 이미지 파일로 저장
plt.show()

# 긍정/부정 빈도수 그래프 그리기
sentiments = list(sentiment_count.keys())
sentiment_counts = list(sentiment_count.values())

# 색상 설정 (colormap 사용)
sentiment_colors = plt.cm.plasma(np.linspace(0, 1, len(sentiments)))  # plasma colormap 사용

plt.figure(figsize=(10, 5))
bars = plt.bar(sentiments, sentiment_counts, color=sentiment_colors)
plt.title('Sentiment Frequency')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')

# 각 막대에 색상과 빈도수 표시
for bar, count in zip(bars, sentiment_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{count}', ha='center', va='bottom')

plt.savefig('sentiment_frequency.png')  # 그래프를 이미지 파일로 저장
plt.show()