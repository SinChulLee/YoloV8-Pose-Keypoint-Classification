# YOLOv8-YogaPose-Keypoint-Classification

## 🏆 Project Introduction
 1. 주제: YoloV8-pose를 이용한 요가 자세 피드백
 2. 팀원: 김기동, 이신철, 조성준, 이요담
 3. 데이터: kaggle opendataset
 4. 사용언어: Python
 5. 개발환경: Jupyter Notebook, Colab
 6. 라이브러리: ultralytics, sklearn, pandas, numpy, matplotlib, seaborn, Pytorch

    




## 📖 Research and Analysis

요가는 5000년 전에 인도에서 기원한 고대의 과학적 수련으로 몸과 마음 간의 조화를 이루는데 아사나, 명상 그리고 다양한 숨쉬기 기술을 활용하고 마음의 평화를 가져온다. 현대 생활에서 스트레스가 증가함에 따라 요가는 전 세계적으로 인기를 얻고 있다. 요가를 배울 수 있는 방법은 다양한데, 요가 센터에서 수업을 듣거나 가정에서 가르침을 받는 것이 그 중 하나이고 책이나 동영상의 도움으로 스스로 학습할 수도 있다. 대부분의 사람들은 자기 학습을 선호하지만, 자신의 요가 자세의 오류를 찾기는 어려울 수도 있다. 이 시스템을 사용하면 사용자가 연습하려는 자세를 선택할 수 있다. 그 다음 사용자는 해당 자세를 취한 사진과 영상을 업로드 할 수 있다. 사용자의 자세는 전문가의 자세와 비교되며 각각의 관절 각도의 차이가 계산된다. 이 각도의 차이를 기반으로 사용자에게 피드백이 제공되어 자세를 개선할 수 있다. 





## 📝 Data Pre-processing

Data pre-processing 과정은 다음과 같다.


###  1. Pydantic
   - 데이터 유효성 검사 및 구조화를 위해 Pydantic을 사용한다.
   - Base model은 타입 힌트를 이용한 자동검증, 기본값 및 필수 필드 지정, 직렬화 및 역직렬화, 모델간 상속 및 확장 기능을 제공해준다. 특히 JSON과 같은 형식으로 변환할 수 있어 유용하다.

###  2. keypoint 추출
   - ultralytics 라이브러리에서 제공하는 YOLO 클래스 인스턴스화하여 YOLO 모델을 관리하고 우리가 설정한 매개변수로 모델을 초기화 하였다.
   - OpenCV를 사용하여 이미지를 읽고, YOLOv8 모델을 사용하여 해당 이미지에서 keypoint 요소를 추출
   - 추출된 keypoint를 index로 mapping
   - mapping된 keypoint에서 예측에 필요한 index들만 사용




## 🏆 Modeling

### Multiple Linear Clasification 
   - 필요한 keypoint들의 index값(머리부분 빼고)들을 가지고 다중분류 적용.



**[참고] Confusion Matrix**
<p align="center">
  <img src="https://github.com/SinChulLee/YoloV8-Pose-Keypoint-Classification/assets/145883892/e34f1f95-ed21-42fb-8214-36db0c415c4a" width="637" height="527">
</p>

