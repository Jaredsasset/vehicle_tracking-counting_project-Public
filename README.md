# Vehicle_tracking-counting_project-Public

## Contents
[💡 Intro](#-introduction)
1. [Project Introduction](#project-introduction)
2. [Project Background](#project-background)

[⚙️ Development Process](#development-process)
1. [Data Description](#1-data-description)
2. [Model](#2-model)
   - YOLOv5 result video
   - YOLOv8x result video
3. [Limitations](#limitations)
--------
## 💡 Introduction

### Project Introduction
“Vehicle Tracking and Counting”
다양한 YOLO 모델을 학습시키고 그 성능을 확인한 후 가장 우수한 모델을 활용하여 Object Counting을 구현 
단위시간마다의 통행량을 데이터 프레임으로 변환하고 이를 csv파일로 저장 및 시각화
지역 특성을 고려한 차종별 통행량 분석 및 전국 평균 통행량과의 비교 분석

### Project Background
- 최근 10년간 교통량과 자동차등록대수가 매해 지속적으로 증가하고 있다는 점과 통행량 밀집도와 도로손상 및 미세먼지 농도가 양의 상관관계를 가지고 있다는 점
- 교통체증으로 인한 시민들의 불편함
- 지역특성에 따라 특정 차종의 통행제한이 이루어지고 있다는 점
  
   해당 object tracking and counting이 차종 및 유입, 유출 통행량 분석을 통해 해당 문제점들을 해결할 수 있는 발판을 제공할 것이라 기대
---------

## Development Process

### 1. Data Description
![dataset](https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/44ace999-b42c-411d-9338-1421175aa0a7)
Train과 Validation을 전체 Dataset을 기준으로 7:3으로 나누었다.

- sample images for each class
![스크린샷 2023-08-11 오전 6 03 24](https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/59f463cb-8ddf-4588-8cb2-6ce61f65eb88)
### 2. Model
Trained YOLOv5, YOLOv7, YOLOv7-tiny, YOLOv8x

####Metrics
![스크린샷 2023-08-10 오후 8 51 15](https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/a0906d33-602b-4f3a-b0cc-136149c5905e)
#### Video result for YOLOv5

https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/684e372a-319e-4994-8602-3b3ff7f0dc39

#### Video result for YOLOv8x (Including object counting)


https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/62ebd047-a192-439e-a840-b9baa773ea97



https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/9ba6f248-2605-44e4-96db-6dac8b3cd726
- 모델 중 mAP50과 mAP50-95를 기준으로 성능이 가장 좋은 2개를 활용하여 object detection에 적용하여 detection 성능을 확인
- YOLO v5 custom trained model은 작은 물체를 잘 감지하지 못하는 반면 pertained YOLO v8x는 멀리 있는 차량도 잘 감지하였다. 따라서 YOLOv8x로 object tracking 및 counting을 구현했다.
#### 3. 시간대별 차량 통행 수 
<img height = '300' src = 'https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/4573de1e-267c-4583-9811-bfbee0580d47'>
<img height = '300' src = 'https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/ceac4c29-5ed1-4025-8968-4d2dc4289845'>

<img height = '300' src = 'https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/7e44d95d-3f8f-48e3-bc27-25884e94c543'>

<img height = '300' src = 'https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/5b4e9580-d2b2-4986-992f-d1f3254c9b0b'>

<img height = '300' src = 'https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/c82a33c9-ece1-4932-929f-bbd376d8b150'>

<img height = '300' src = 'https://github.com/Jaredsasset/vehicle_tracking-counting_project-Public/assets/132141381/4885424c-110f-4b3e-b7d8-c7dceb021daf'>

##### 차종별 전국 평균 일교통량과 크게 다르지 않다.
##### 교차로 지역에 트럭의 수가 평균에 비해 약 30%인 이유는 해당 지역이 어린이보호구역이기 때문인 것으로 추정했다.
---------

## Limitations
- YOLOv5의 경우 학습시킨 데이터가 차량을 근접거리에서 촬영한 이미지였기 때문에 차량이 CCTV와 거리가 멀어지면 감지 성능이 떨어지는 문제가 있었다.
- CCTV영상 데이터를 프레임별로 나누고 이를 바탕으로 레이블링을 해서 학습을 시키면 감지성능이 개선될 것으로 생각된다.
- 동일한 지역에서 다양한 시간대의 데이터가 존재했더라면 데이터 분석을 더 신빙성 있게 할 수 있을 것이다.

<a href="https://github.com/Jaredsasset"><img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=GitHub&logoColor=white"/> <a href="https://velog.io/@fnrfn2"><img src="https://img.shields.io/badge/Blog-20C997?style=flat-square&logo=Velog&logoColor=white"/> <a href="mailto:fnffn2354@gmail.com"><img src="https://img.shields.io/badge/Mail-EA4335?style=flat-square&logo=Gmail&logoColor=white"/></a>|

















