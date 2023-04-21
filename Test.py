import cv2

img1 = cv2.imread('Test3.png')
img2 = cv2.imread('Test2.jpg')

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

_, thresh_black1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
_, thresh_black2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

contours_black1, _ = cv2.findContours(thresh_black1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_black2, _ = cv2.findContours(thresh_black2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

num_black_squares1 = 0
num_black_squares2 = 0

# Loop through each contour for the green squares in the first image
for cnt1 in contours_black1:
    perimeter1 = cv2.arcLength(cnt1, True)
    approx1 = cv2.approxPolyDP(cnt1, 0.02*perimeter1, True)
    if len(approx1) == 4 and cv2.contourArea(cnt1) > 1000 and cv2.contourArea(cnt1) < 10000:
        num_black_squares1 += 1

# Loop through each contour for the green squares in the second image
for cnt2 in contours_black2:
    perimeter2 = cv2.arcLength(cnt2, True)
    approx2 = cv2.approxPolyDP(cnt2, 0.02*perimeter2, True)
    if len(approx2) == 4 and cv2.contourArea(cnt2) > 1000 and cv2.contourArea(cnt2) < 10000:
        num_black_squares2 += 1

# Print the number of green squares in both images
print(f"Number of green squares in Image from 3 months prior: {num_black_squares1}")
print(f"Number of green squares in Current image: {num_black_squares2}")

# Compare the differences
if num_black_squares1 > num_black_squares2:
    print("Image from 3 months prior has more green squares.")
    print("The differences is:", (num_black_squares1 - num_black_squares2))
elif num_black_squares1 < num_black_squares2:
    print("Current image has more green squares.")
    print("The differences is:", (num_black_squares2 - num_black_squares1))
else:
    print("Both images have the same number of green squares.")

#Outline the image using red line
cv2.drawContours(img1, contours_black1, -1, (0, 0, 255), 2)
cv2.drawContours(img2, contours_black2, -1, (0, 0, 255), 2)

cv2.imshow('Image from 3 months prior', img1)
cv2.imshow('Current image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()