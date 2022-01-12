import cv2 as cv

file = 'totoro.jpg'
background = cv.imread('background.png')

img = cv.imread(file)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

mask = cv.inRange(hsv, (50, 150, 0),(70, 255, 255))
#  OpenCV에서 제공하는 cv2.inRange 함수를 사용하여 특정 생삭 영역을 추출할 수 있습니다.

cv.copyTo(background, mask, img)
# cv2.copyTo()함수를 이용하여 마스크 연산을 해보도록 하겠습니다.

cv.imshow('result', img)
cv.waitKey()