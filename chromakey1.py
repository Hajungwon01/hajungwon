import cv2 as cv
clicked = False

background = cv.imread('background.png')

def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv.EVENT_LBUTTONUP:
        clicked = True


cameraCapture = cv.VideoCapture(0, cv.CAP_DSHOW)         # 0은 비디오 입력 장치의 번호이므로 바뀔 수도 있음.
cv.namedWindow('MyWindow')         # 미리 출력할 창을 생성해 둔다.
cameraCapture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cameraCapture.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

success = True
while success and cv.waitKey(1) == -1 and not clicked: # 키 입력이 없으면 and 마우스 left button 놓은 적이 없으면
    success, frame = cameraCapture.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, (-10, 100, 100), (10, 255, 255))
    cv.copyTo(background, mask, frame)

    cv.imshow('MyWindow', frame)

cv.destroyWindow('MyWindow')
cameraCapture.release()