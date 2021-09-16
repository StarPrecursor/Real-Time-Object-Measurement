import cv2
import numpy as np


def adjust_img(img, brightness=30, contrast=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # lim = 255 - value
    # v[v > lim] = 255
    # v[v <= lim] += value

    v = v * (contrast / 127 + 1) - contrast + brightness
    v = np.clip(v, 0, 255)
    v = np.uint8(v)

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    (b, g, r) = cv2.split(img.astype("float"))
    g = 1.5 * g
    b = b * 0.2
    r = r * 0.2
    g = np.clip(g, 0, 255)
    print("b shape", b.shape)
    img = cv2.merge((b, g, r))
    img = np.uint8(img)

    return img


def getContours(
    img, cThr=[100, 100], showCanny=False, minArea=1000, filter=0, draw=False
):
    ori_img = np.copy(img)
    img = adjust_img(img, brightness=80, contrast=120)
    cv2.imshow("Bright", img)
    #imgGray = cv2.cvtColor(imgBright, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 1)
    cv2.imshow("Blur", img)
    img = cv2.Canny(img, cThr[0], cThr[1])
    kernel = np.ones((3, 3))
    img = cv2.dilate(img, kernel, iterations=1)
    #img = cv2.erode(img, (1, 1), iterations=5)

    line_img = np.copy(img) * 0
    lines = cv2.HoughLinesP(img, 1, np.pi / 3600, 15, np.array([]), 50, 20)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 0, 0), 5)
    pad = np.zeros(line_img.shape)
    print(line_img.shape, pad.shape)
    line_img = cv2.merge((line_img, line_img, line_img))
    print(ori_img.shape, line_img.shape)
    img_l = cv2.addWeighted(ori_img, 0.8, line_img, 1, 0)

    if showCanny:
        cv2.imshow("Canny", img_l)

    contours, hiearchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(approx)
            if filter > 0:
                if len(approx) == filter:
                    finalCountours.append((len(approx), area, approx, bbox, i))
            else:
                finalCountours.append((len(approx), area, approx, bbox, i))
    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)

    return img, finalCountours
    # cv2.findContours(imgThre, )

def reorder(myPoints):
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4, 2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew

def warpImg(img, points, w, h, pad=20):
    #print(points)
    new_points = reorder(points)
    #print(new_points)

    pts1 = np.float32(new_points)
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgWarp = cv2.warpPerspective(img, matrix, (w, h))
    imgWarp = imgWarp[pad: imgWarp.shape[0] - pad, pad:imgWarp.shape[1] - pad]
    
    return imgWarp

def findDis(pts1, pts2):
    return ((pts2[0] - pts1[0]) ** 2 + (pts2[1] - pts1[1]) ** 2) ** 0.5

