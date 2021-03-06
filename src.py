import cv2 as cv
import time
import numpy as np

# position 트랙바 콜백 함수
def callback_P(x):
    videoCapture.set(cv.CAP_PROP_POS_FRAMES, x)
# =====================================================

# HE wgt 트랙바 콜백 함수
def callback_W(x):
    pass
# =====================================================
# 시그마 트랙바 콜백 함수
def callback_S(x):
    pass
# scale 트랙바 콜백 함수
def callback_S1(x):
    pass

def UM1(src):
    src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
    src_f = src_ycrcb[:, :, 0].astype(np.float32)  # split 대신 슬라이싱 사용
    blr = cv.GaussianBlur(src_f, (0, 0), cv.getTrackbarPos('sigma', 'Video quality improvement editing program'))
    src_ycrcb[:, :, 0] = np.clip(src_f + cv.getTrackbarPos('scale', 'Video quality improvement editing program')*(src_f - blr), 0, 255).astype(np.uint8)  # 중간 연산은 실수가 좋고 최종 형태는 unit8로 하는게 좋음
    dst = cv.cvtColor(src_ycrcb, cv.COLOR_YCrCb2BGR)
    return dst


def UM(img):
    k = cv.getTrackbarPos('sigma', 'Video quality improvement editing program') * 6 + 1
    blur = cv.GaussianBlur(src=img, ksize=(k, k),sigmaX=cv.getTrackbarPos('sigma', 'Video quality improvement editing program'))
    UnsharpMaskImg = img - blur
    SharpenedImg = img + cv.getTrackbarPos('scale', 'Video quality improvement editing program') * UnsharpMaskImg
    return SharpenedImg

def HE(img):
    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist, bins = np.histogram(imgG, 256, [0, 255])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()
    per = float(cv.getTrackbarPos('HE wgt', 'Video quality improvement editing program')) / 100
    value = (255/cdf[255]) * per
    mapping = cdf * value
    LUT = np.clip(mapping, 0, 255).astype('uint8')
    imgCeq = LUT[img]
    #dst = cv.cvtColor(imgCeq, cv.COLOR_GRAY2BGR)
    return imgCeq


# 파일 지정
Path = "../data/"
Name = 'matrix.avi'

FullName = Path + Name
SaveFileName = 'test_' + Name # 저장할 파일 이름
# ======================================================

# 읽기용 객체 생성
videoCapture = cv.VideoCapture(FullName)
# =======================================================

# 동영상의 정보
number_of_total_frames = videoCapture.get(cv.CAP_PROP_FRAME_COUNT)
fps = videoCapture.get(cv.CAP_PROP_FPS)
size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH))*2,
    int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
dly_ms = 1000/(fps)
# ========================================================

# 창 만들기
cv.namedWindow('Video quality improvement editing program')
# =========================================================

# 트랙바 지정
# 1) 위치
cv.createTrackbar ('position',#트랙바 앞에 표시될 트랙바의 이름
'Video quality improvement editing program',#트랙바가 나타날 창의 이름
0,#시작 당시의 슬라이더의 초기 위치
int(number_of_total_frames), callback_P)#슬라이더가 움직일 때 호출될 콜백 함수의 이름.
                    #첫 번째 파라미터:트랙 바 위치.두 번째 파라미터:사용자 데이터.
# 2) HE 가중치
cv.createTrackbar ('HE wgt',#트랙바 앞에 표시될 트랙바의 이름
'Video quality improvement editing program',#트랙바가 나타날 창의 이름
50,#시작 당시의 슬라이더의 초기 위치
100, callback_W)#슬라이더가 움직일 때 호출될 콜백 함수의 이름.
                    #첫 번째 파라미터:트랙 바 위치.두 번째 파라미터:사용자 데이터.
# 3) sigma
cv.createTrackbar ('sigma',#트랙바 앞에 표시될 트랙바의 이름
'Video quality improvement editing program',#트랙바가 나타날 창의 이름
1,#시작 당시의 슬라이더의 초기 위치
8, callback_S)#슬라이더가 움직일 때 호출될 콜백 함수의 이름.
                    #첫 번째 파라미터:트랙 바 위치.두 번째 파라미터:사용자 데이터.
# 3) scale
cv.createTrackbar ('scale',#트랙바 앞에 표시될 트랙바의 이름
'Video quality improvement editing program',#트랙바가 나타날 창의 이름
1,#시작 당시의 슬라이더의 초기 위치
6, callback_S1)#슬라이더가 움직일 때 호출될 콜백 함수의 이름.
                    #첫 번째 파라미터:트랙 바 위치.두 번째 파라미터:사용자 데이터.

# 비디오 쓰기용 객체 생성
CODEC = cv.VideoWriter_fourcc('F', 'M', 'P', '4')
videoWriter = cv.VideoWriter(
SaveFileName, CODEC, fps, size, isColor=True)         #정상 속도.똑 같은 파일을 만든다.
# =========================================================

font = 	cv.FONT_HERSHEY_DUPLEX

# 비디오 재생 및 트랙바 적용, 비디오 저장, 이미지 캡쳐
success, frame = videoCapture.read()

count = 0
margin = 1      # 순수한 영상출력(재생) 외의 다른 작업에 소비되는 예상 추정시간[ms]. 경험치

s_time = time.time()            # ms 단위의 현재 tick count을 반환
while success:          # Loop until there are no more frames.
    s = time.time()     # start. time in sec.
    frame1 = frame.copy()
    frame2 = HE(frame1)

    if cv.getTrackbarPos('sigma', 'Video quality improvement editing program') != 0 and cv.getTrackbarPos('scale', 'Video quality improvement editing program') != 0:
        frame3 = UM1(frame2)


    cv.putText(frame3, f"sigma = {cv.getTrackbarPos('sigma', 'Video quality improvement editing program')}, scale = {cv.getTrackbarPos('scale', 'Video quality improvement editing program')}, weight = {cv.getTrackbarPos('HE wgt', 'Video quality improvement editing program')}",
               (5, int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT))-10), font, 0.5, (0, 0, 155), 1)
    cv.putText(frame1, "Original", (5, 15), font, 0.5, (0, 0, 155), 2)
    shw = np.hstack((frame1, frame3))

    cv.imshow('Video quality improvement editing program', shw)
    videoWriter.write(shw)
    cv.setTrackbarPos("position", "Video quality improvement editing program", int(videoCapture.get(cv.CAP_PROP_POS_FRAMES)))
    k = cv.waitKey(1)  # & 0xFF
    if k == ord('s'):  # s키를 입력할 경우, 해당 장면을 이미지 캡쳐 저장
        cv.imwrite('test.jpg', np.hstack((frame1, frame3)))
    elif k != -1:  # 그 외의 키는 비디오 중지
        break
    success, frame = videoCapture.read()
    count = count + 1
    print("\rCurrent frame number = ", count, end=' ')



    while ( (time.time() - s) * 1000 ) < (dly_ms - margin): # dly_ms: ms로 표시한 프레임간의 간격[ms]
        pass

videoWriter.release()
videoCapture.release()
# =========================================================