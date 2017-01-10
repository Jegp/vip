import cv2

img = cv2.imread('lenna.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img2=cv2.drawKeypoints(gray,kp,img)


cv2.imshow('image',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()