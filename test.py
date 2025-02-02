import cv2
import test_utils

###################################
webcam = True
path = "1.jpg"
# path = '2.png'
# path = '3.png'
# path = '4.png'
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("video.mp4")

cap.set(10, 160)
cap.set(3, 1920)
cap.set(4, 1080)
scale = 3
wP = 210 * scale
hP = 297 * scale
###################################

while True:
    if webcam:
        success, img = cap.read()
    else:
        img = cv2.imread(path)
    img = cv2.resize(img, (0, 0), None, 0.5, 0.5)

    imgContours, conts = test_utils.getContours(
        img, showCanny=True, draw=True, minArea=50000, filter=4
    )
    if len(conts) != 0:
        biggest = conts[0][2]
        # print(biggest)
        imgWarp = test_utils.warpImg(img, biggest, wP, hP)
        # cv2.imshow("A4", imgWarp)
        imgContours2, conts2 = test_utils.getContours(
            imgWarp, minArea=2000, filter=4, cThr=[20, 200], draw=False
        )

        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = test_utils.reorder(obj[2])
                nW = test_utils.findDis(nPoints[0][0] / scale, nPoints[1][0] / scale)
                nW = round(nW, 2)
                nH = test_utils.findDis(nPoints[0][0] / scale, nPoints[2][0] / scale)
                nH = round(nH, 2)
                cv2.arrowedLine(
                    imgContours2,
                    (nPoints[0][0][0], nPoints[0][0][1]),
                    (nPoints[1][0][0], nPoints[1][0][1]),
                    (255, 0, 255),
                    3,
                    8,
                    0,
                    0.05,
                )
                cv2.arrowedLine(
                    imgContours2,
                    (nPoints[0][0][0], nPoints[0][0][1]),
                    (nPoints[2][0][0], nPoints[2][0][1]),
                    (255, 0, 255),
                    3,
                    8,
                    0,
                    0.05,
                )
                x, y, w, h = obj[3]
                cv2.putText(
                    imgContours2,
                    "{}mm".format(nW),
                    (x + 30, y - 10),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    imgContours2,
                    "{}mm".format(nH),
                    (x - 70, y + h // 2),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5,
                    (255, 0, 255),
                    2,
                )
        cv2.imshow("A4", imgContours2)

    cv2.imshow("Original", img)
    cv2.waitKey(50)
