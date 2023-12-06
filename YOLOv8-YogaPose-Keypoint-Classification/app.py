# Import library
import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

from src.detection_keypoint import DetectKeypoint
from src.classification_keypoint import KeypointClassification
from src.calc_degree import calc_degree

# 폰트설정(폰트객체 생성)
font_path = r"C:\Users\Playdata\Desktop\미니프로젝트_상반기 자료\miniproject\YoloV8-Pose-Keypoint-Classification\NanumBarunGothicBold.ttf"
font = ImageFont.truetype(font_path, 20) # 폰트파일경로, 폰트크기



detection_keypoint = DetectKeypoint()
classification_keypoint = KeypointClassification(
    r"C:\Users\Playdata\Desktop\미니프로젝트_상반기 자료\miniproject\YoloV8-Pose-Keypoint-Classification\models\pose_classification.pth"
)


# 자세에 따른 각도 계산 및 피드백 출력 함수
# labels -> ['Warrior2', 'Plank', 'Goddess', 'Tree', 'Downdog']
# posename = results_classification
# keypoints = results.keypoints
def TextExtract(posename, keypoints):

  # Warrior2 자세일 경우 -> 수정 완료
  if posename == 'Warrior2':
    # 세 점 사이 각도 구하는 함수. 상체 양쪽 팔 쭉 피도록 한다. keypoint index: 오른쪽 손목-10, 왼쪽 어깨-5, 왼쪽 손목-9
    degree1 = calc_degree(keypoints.xy[0, 10].cpu().numpy(), keypoints.xy[0, 5].cpu().numpy(), keypoints.xy[0, 9].cpu().numpy())

    # 왼쪽다리 keypoint index: 왼쪽 엉덩이-11, 왼쪽 무릎:13, 왼쪽 발목-15
    degree3 = calc_degree(keypoints.xy[0, 11].cpu().numpy(), keypoints.xy[0, 13].cpu().numpy(), keypoints.xy[0, 15].cpu().numpy())

    # 오른쪽 다리 keypoint index: 오른쪽 엉덩이-12, 오른쪽 무릎:14, 오른쪽 발목-16
    degree4 = calc_degree(keypoints.xy[0, 12].cpu().numpy(), keypoints.xy[0, 14].cpu().numpy(), keypoints.xy[0, 16].cpu().numpy())

    feedback_text = ""
    # text = f"{degree1}\n{degree3}\n{degree4}\n" # 각 부위의 각도 확인 텍스트

    # 상체 피드백
    if 160 <= degree1 <= 200:
      feedback_text += "올바른 자세입니다.(상체)\n"
    else:
      feedback_text += "팔을 수평으로 맞춰주세요\n"

    # 다리 피드백
    # 양쪽 다리의 각도를 구하여 어느 다리가 굽혔는지 결정
    if degree3 > degree4: # 오른쪽 다리가 굽혔을 때 -> 오른쪽 다리에 대한 Feedback
      if degree4 > 120:
        feedback_text += "오른쪽 다리의 각도를 좁혀주세요."
      elif degree4 < 50:
        feedback_text += "오른쪽 다리의 각도를 넓혀주세요."
      else:
        feedback_text += "올바른 자세입니다.(하체)"
    else: # 왼쪽 다리가 굽혔을 때 -> 오른쪽 다리에 대한 Feedback
      if degree3 > 120:
        feedback_text += "왼쪽 다리의 각도를 좁혀주세요."
      elif degree3 < 50:
        feedback_text += "왼쪽 다리의 각도를 넓혀주세요."
      else:
        feedback_text += "올바른 자세입니다.(하체)"

    return feedback_text

  # Plank 자세일 경우 -> 수정 완료
  elif posename == 'Plank':
    # Plank 자세의 특성상 측면으로 사진을 찍기 때문에 팔의 각도를 구할 때 왼쪽이나 오른쪽으로 고정한다면, keypoint의 좌표가 인식되지 않아 [0,0]이 되어 각도를 잘못 구할 가능성이 있다.
    # 그래서 한쪽 팔의 keypoint 좌표 중 하나라도 0이 있다면 반대쪽 팔의 keypoint 좌표를 이용하여 각도를 계산한다.
    
    # 왼쪽 추출 keypoints 중 하나라도 0이 있다면, 오른쪽 keypoint 추출 조건식
    if keypoints.xy[0, 5].cpu() == [0, 0] or keypoints.xy[0, 11].cpu() == [0, 0] or keypoints.xy[0, 15].cpu() == [0, 0]:
      # keypoint index: 오른쪽 어깨-6, 오른쪽 골반:12, 오른쪽 발목-16
      degree = calc_degree(keypoints.xy[0, 6].cpu().numpy(), keypoints.xy[0, 12].cpu().numpy(), keypoints.xy[0, 16].cpu().numpy())
    else:
      # 세 점 사이 각도 구하는 함수. keypoint index: 왼쪽 어깨-5, 왼쪽 골반:11, 왼쪽 발목-15
      degree = calc_degree(keypoints.xy[0, 5].cpu().numpy(), keypoints.xy[0, 11].cpu().numpy(), keypoints.xy[0, 15].cpu().numpy())
    
    # text = f"{degree}\n" # 각도 테스트

    feedback_text = ""
    
    # 몸통 피드백 : 몸이 일직선으로 펴져 있는지 체크 및 Feedback
    if 160 <= degree <= 200:
      feedback_text += f"올바른 자세입니다."
    else:
      feedback_text += "몸을 수평하게 만들어 주세요."

    return feedback_text

  # Goddess 자세일 경우 -> 수정 완료
  elif posename == 'Goddess':
    # Goddess자세는 허리 위로 (팔부분)이 너무 다양하여 다리의 각도만 사용해서 피드백하기로 결정.

    # 왼쪽다리 keypoint index: 왼쪽 엉덩이-11, 왼쪽 무릎:13, 왼쪽 발목-15
    degree1 = calc_degree(keypoints.xy[0, 11].cpu().numpy(), keypoints.xy[0, 13].cpu().numpy(), keypoints.xy[0, 15].cpu().numpy())

    # 오른쪽다리 keypoint index: 오른쪽 엉덩이-12, 오른쪽 무릎:14, 오른쪽 발목-16
    degree2 = calc_degree(keypoints.xy[0, 12].cpu().numpy(), keypoints.xy[0, 14].cpu().numpy(), keypoints.xy[0, 16].cpu().numpy())

    degree_list = [degree1, degree2]
    feedback_text = ""
    
    # 양쪽 다리에 대한 Feedback -> 다리가 적당히 벌려져 있다면(70도 ~ 130도) 올바른 자세이다. 
    if 70 <= degree1 <= 130 and 70 <= degree2 <= 130:
      feedback_text += "올바른 자세 입니다."
    else:
      for i in degree_list:
        if i == degree1:
          if i < 70:
            feedback_text += "왼쪽다리를 더 펴주세요\n"
          elif i > 130:
            feedback_text += "왼쪽다리를 굽혀주세요\n"
        else:
          if i < 70:
            feedback_text += "오른다리를 더 펴주세요\n"
          elif i > 130:
            feedback_text += "오른다리를 굽혀주세요\n"

    return feedback_text

  # Tree 자세일 경우 -> 수정완료
  elif posename == 'Tree':

    # 왼쪽다리각도 : 왼쪽 골반:11, 왼쪽 무릎-13, 왼쪽 발목-15
    degree1 = calc_degree(keypoints.xy[0, 11].cpu().numpy(), keypoints.xy[0, 13].cpu().numpy(), keypoints.xy[0, 15].cpu().numpy())
    # 오른쪽다리각도 : 오른쪽 골반-12, 오른쪽 무릎:14, 오른쪽 발목-16
    degree2 = calc_degree(keypoints.xy[0, 12].cpu().numpy(), keypoints.xy[0, 14].cpu().numpy(), keypoints.xy[0, 16].cpu().numpy())
    
    feedback_text = "" # 피드백 텍스트
    
    # Tree 자세의 특성상 다리 하나는 무조건 올리기 때문에 어느 다리를 올렸는지 판별 후 Feedback
    if degree1 > degree2: # 오른발 올림
      if degree2 > 100: # 다리가 많이 내려가 있을 경우
        feedback_text += "오른발을 더 올려주세요.\n"
      else:
        feedback_text += "올바른 자세입니다.\n"
    else: # 왼발 올림
      if degree1 > 100: # 다리가 많이 내려가 있을 경우
        feedback_text += "왼발을 더 올려주세요.\n"
      else:
        feedback_text += "올바른 자세입니다.\n"

    return feedback_text

  # Downdog 자세일 경우 -> 수정 완료
  elif posename == 'Downdog':
    # Downdog 자세도 Plank 자세와 마찬가지로 자세의 특성상 측면으로 사진을 찍기 때문에 팔의 각도를 구할 때 왼쪽이나 오른쪽으로 고정한다면, keypoint의 좌표가 인식되지 않아 [0,0]이 되어 각도를 잘못 구할 가능성이 있다.
    # 그래서 한쪽 부분의 keypoint 좌표 중 하나라도 0이 있다면 반대쪽 부분의 keypoint 좌표를 이용하여 각도를 계산한다.
    
    # 왼쪽 추출 keypoints 중 하나라도 0이 있다면, 오른쪽 keypoint 추출 조건식
    if keypoints.xy[0, 5].cpu() == [0, 0] or keypoints.xy[0, 9].cpu() == [0, 0] or keypoints.xy[0, 11].cpu() == [0, 0] or keypoints.xy[0, 15].cpu() == [0, 0]:
      # keypoint index: 오른쪽 어깨-6, 오른쪽 골반-12, 오른쪽 발목-16
      degree1 = calc_degree(keypoints.xy[0, 6].cpu().numpy(), keypoints.xy[0, 12].cpu().numpy(), keypoints.xy[0, 16].cpu().numpy())
      # keypoint index: 오른쪽 손목-10, 오른쪽 어깨-6, 오른쪽 골반-12
      degree2 = calc_degree(keypoints.xy[0, 10].cpu().numpy(), keypoints.xy[0, 6].cpu().numpy(), keypoints.xy[0, 12].cpu().numpy())
    else:
      # keypoint index: 왼쪽 어깨-5, 왼쪽 골반-11, 왼쪽 발목-15
      degree1 = calc_degree(keypoints.xy[0, 5].cpu().numpy(), keypoints.xy[0, 11].cpu().numpy(), keypoints.xy[0, 15].cpu().numpy())
      # keypoint index: 왼쪽 손목-9, 왼쪽 어깨-5, 왼쪽 골반-11
      degree2 = calc_degree(keypoints.xy[0, 9].cpu().numpy(), keypoints.xy[0, 5].cpu().numpy(), keypoints.xy[0, 11].cpu().numpy())
    
    degree_list = [degree1, degree2]
    feedback_text = ""

    # Text = f"{degree1}\n {degree2}\n"
    if 70 <= degree1 <= 120 and 150 <= degree2 <= 200:
      feedback_text += "올바른 자세입니다."
    else:
      for i in degree_list:
        if i == degree1: 
            if degree1 < 70: # 어깨,골반,발목의 각도가 70도 보다 작다는 의미는 엉덩이가 너무 올라갔다는 의미이므로 손과 발 사이의 공간을 확보해야 한다는 Feedback이 들어가야 한다.
                feedback_text += "손과 발 사이를 넓혀주세요.\n"
            elif degree1 > 120:
                feedback_text += "손과 발 사이를 좁혀주세요.\n"
        else:
            if degree2 < 150 or degree2 > 200: # 손목,어깨,골반의 각도가 수평(180도)과 가까이 되지 않는다면 올바른 자세가 아니므로 Feedback
                feedback_text += "상체를 수평이 되도록 만들어주세요."

    return feedback_text

def pose_classification(img, col=None):
    image = Image.open(img)
    image = image.resize((1080,720))
    
    # 실행, 키포인트 추출
    results = detection_keypoint(image)
    # 키포인트 값
    results_keypoints = results.keypoints
    
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    # show image col 1
    col1.write("Original Image :")
    col1.image(image_rgb)

    # detection keypoint
    results = detection_keypoint(image_cv)
    results_keypoint = detection_keypoint.get_xy_keypoint(results)

    # classification keypoint
    input_classification = results_keypoint[10:]
    results_classification = classification_keypoint(input_classification)

    # visualize result
    image_draw = results.plot(boxes=False)
    x_min, y_min, x_max, y_max = results.boxes.xyxy[0].numpy()
    image_draw = cv2.rectangle(
                    image_draw, 
                    (int(x_min), int(y_min)),(int(x_max), int(y_max)), 
                    (0,0,255), 2
                )
    (w, h), _ = cv2.getTextSize(
                    results_classification.upper(), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
    image_draw = cv2.rectangle(
                    image_draw, 
                    (int(x_min), int(y_min)-20),(int(x_min)+w, int(y_min)), 
                    (0,0,255), -1
                )
    image_draw = cv2.putText(image_draw,
                    f'{results_classification.upper()}',
                    (int(x_min), int(y_min)-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255),
                    thickness=2
                )
    # 텍스트 설정
    text = TextExtract(results_classification, results_keypoints) # 자세 및 각도 파라미터를 함수로 받아 출력할 텍스트를 return 받는다.
    img_pil = Image.fromarray(image_draw)
    img_pil = img_pil.resize((1080,720))
    draw = ImageDraw.Draw(img_pil)
    # (좌표, 좌표): 좌하단좌표 값
    draw.text((800,50), text, (0, 0, 0), font=font)
    image_draw = np.array(img_pil)
    
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)
    col2.write("Keypoint Result :wrench:")
    col2.image(image_draw)
    col2.text(f'Pose Classification : {results_classification}')
    return image_draw, results_classification

st.set_page_config(
    layout="wide", 
    page_title="YoloV8 Keypoint Classification"
)
st.write(
    "## Yolov8을 활용한 요가 자세 분류 및 피드백"
)
st.write(
    ":dog: Downdog, Goddess, Plank, Tree, Warrior2 같은 대표적인 요가 자세를 업로드 해주세요.:grin:"
)
st.sidebar.write(
    "## 사진 업로드 :gear:"
)

col1, col2 = st.columns(2)
img_upload = st.sidebar.file_uploader("사진을 업로드해주세요", type=["png", "jpg", "jpeg"])

if img_upload is not None:
    pose_classification(img=img_upload)

# show sample image
st.write('## 예시')
images = glob.glob('./images/*.jpeg')
row_size = len(images)
grid = st.columns(row_size)
col = 0
for image in images:
    with grid[col]:
        st.image(f'{image}')
        file_name_with_extension = os.path.basename(image)
        file_name_without_extension = os.path.splitext(file_name_with_extension)[0]
        st.subheader(f"{file_name_without_extension}")
        st.button(label='보기', key=f'run_{image}',
                  on_click=pose_classification, args=(image, 'run'))
    col = (col + 1) % row_size

