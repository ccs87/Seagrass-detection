import cv2

cap = cv2.VideoCapture(0)

ret, img1 = cap.read()
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

while True:
    ret, img2 = cap.read()
    if not ret:
        break

    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    _, thresh_black1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, thresh_black2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    contours_black1, _ = cv2.findContours(thresh_black1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_black2, _ = cv2.findContours(thresh_black2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    num_green_squares1 = 0
    num_green_squares2 = 0
    num_white_squares1 = 0
    num_white_squares2 = 0

    for cnt1 in contours_black1:
        perimeter1 = cv2.arcLength(cnt1, True)
        approx1 = cv2.approxPolyDP(cnt1, 0.02*perimeter1, True)
        area1 = cv2.contourArea(cnt1)
        if len(approx1) == 4 and area1 > 1000 and area1 < 10000:
            x, y, w, h = cv2.boundingRect(cnt1)
            square_roi = gray1[y:y+h, x:x+w]
            if cv2.mean(square_roi)[0] < 128:
                num_green_squares1 += 1
            else:
                num_white_squares1 += 1

    for cnt2 in contours_black2:
        perimeter2 = cv2.arcLength(cnt2, True)
        approx2 = cv2.approxPolyDP(cnt2, 0.02*perimeter2, True)
        area2 = cv2.contourArea(cnt2)
        if len(approx2) == 4 and area2 > 1000 and area2 < 10000:
            x, y, w, h = cv2.boundingRect(cnt2)
            square_roi = gray2[y:y+h, x:x+w]
            if cv2.mean(square_roi)[0] < 128:
                num_green_squares2 += 1
            else:
                num_white_squares2 += 1

    print(f"Number of green squares in previous frame: {num_green_squares1}")
    print(f"Number of green squares in current frame: {num_green_squares2}")
    print(f"Number of white squares in previous frame: {num_white_squares1}")
    print(f"Number of white squares in current frame: {num_white_squares2}")

    if num_green_squares1 > num_green_squares2:
        print("Previous frame has more green squares.")
        print("The difference is:", (num_green_squares1 - num_green_squares2))
    elif num_green_squares1 < num_green_squares2:
        print("Current frame has more green squares.")
        print("The difference is:", (num_green_squares2 - num_green_squares1))
    else:
        print("Both frames have the same number of green squares.")

    if num_white_squares1 > num_white_squares2:
        print("Previous frame has more white squares.")
        print("The difference is:", (num_white_squares1 - num_white_squares2))
    elif num_white_squares1 < num_white_squares2:
        print("Current frame has more white squares.")
        print("The difference is:", (num_white_squares2 - num_white_squares1))
    else:
        print("Both frames have the same number of white squares.")

    cv2.drawContours(img1, contours_black1, -1, (0, 0, 255), 2)
    cv2.drawContours(img2, contours_black2, -1, (0, 0, 255), 2)

    cv2.imshow('Previous frame', img1)
    cv2.imshow('Current frame', img2)
    if cv2.waitKey(1) == ord('q'):
        break

    gray1 = gray2.copy()
    img1 = img2.copy()

cap.release()
cv2.destroyAllWindows()
