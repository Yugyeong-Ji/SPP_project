#  'AI기반 MBTI 예측 웹서비스 프로젝트' in University of Southern California
## 2023 USC 한미 해커톤 2등 수상작

### 텍스트 데이터를 이용한 MBTI 예측 알고리즘

- TF-IDF Vectorizer+GridSearchCV+XGBoost
  : 클래스 불균형이 있기 때문에 GridSearchCV의 'scoring' 매개변수를 'f1_weighted'으로 설정. 이를 통해, Grid Search 과정에서 모델의 성능을 F1-score로 평가하도록 지정
- RNN
  : 클래스 불균형을 해결해보고자, 가중치 조정을 시도했지만 정확도가 오히려 하락해 파라미터 튜닝만 시도
- SMOTE+GridSearchCV+TF-IDF Vectorizer+LinearSVC
  : SMOTE를 이용하여 오버샘플링, Grid Search를 통해 최적의 c값을 찾고 TF-IDF로 벡터화 및 LinearSVC로 학습

|  | XGBoost | RNN | LinearSVC |
| :---- | ------ | :----------: | --------------------: |
| accuracy | 0.50 | 0.59 | 0.967 |
| F1-score | 0.48 | 0.40 | 0.966 |

최종 완성 코드 : MBTIgram_SMOTE+LinearSVC.ipynb
