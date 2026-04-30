import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Read images
img1 = cv.imread(r"C:\Users\Administrator\Desktop\UOM\Sem3\Computer Vision - Dr Ranga\Assignment 02\c1.jpg")
img2 = cv.imread(r"C:\Users\Administrator\Desktop\UOM\Sem3\Computer Vision - Dr Ranga\Assignment 02\c2.jpg")

if img1 is None or img2 is None:
    raise FileNotFoundError("One or both images not found. Check the file paths.")

img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# (a) Manually click 6 corresponding points in each image

pts1 = []
pts2 = []

def click_points_img1(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(pts1) < 6:
            pts1.append((x, y))
            cv.circle(img1_display, (x, y), 5, (0, 0, 255), -1)
            cv.putText(img1_display, str(len(pts1)), (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.imshow("Image 1 - Click 6 Points", img1_display)
            print(f"Image 1 - Point {len(pts1)}: ({x}, {y})")
        if len(pts1) == 6:
            print("Image 1 done. Press any key to continue.")

def click_points_img2(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        if len(pts2) < 6:
            pts2.append((x, y))
            cv.circle(img2_display, (x, y), 5, (0, 0, 255), -1)
            cv.putText(img2_display, str(len(pts2)), (x + 8, y - 8),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.imshow("Image 2 - Click 6 Points", img2_display)
            print(f"Image 2 - Point {len(pts2)}: ({x}, {y})")
        if len(pts2) == 6:
            print("Image 2 done. Press any key to continue.")

# Click points on Image 1
img1_display = img1.copy()
cv.imshow("Image 1 - Click 6 Points", img1_display)
cv.setMouseCallback("Image 1 - Click 6 Points", click_points_img1)
print("Click 6 points on Image 1...")
cv.waitKey(0)
cv.destroyAllWindows()

# Click points on Image 2
img2_display = img2.copy()
cv.imshow("Image 2 - Click 6 Points", img2_display)
cv.setMouseCallback("Image 2 - Click 6 Points", click_points_img2)
print("Click 6 points on Image 2...")
cv.waitKey(0)
cv.destroyAllWindows()

if len(pts1) < 6 or len(pts2) < 6:
    raise ValueError("You must click exactly 6 points in each image.")

pts1 = np.float32(pts1)
pts2 = np.float32(pts2)

# Compute homography from image 1 to image 2
H_manual, mask_manual = cv.findHomography(pts1, pts2)

# Warp image 1 to match image 2
warped_manual = cv.warpPerspective(img1, H_manual, (img2.shape[1], img2.shape[0]))

# (b) Difference image for manual homography

diff_manual = cv.absdiff(warped_manual, img2)
diff_manual_gray = cv.cvtColor(diff_manual, cv.COLOR_BGR2GRAY)


# (c) ORB keypoints, descriptors, and matching

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

orb = cv.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

match_img = cv.drawMatches(
    img1_rgb, kp1,
    img2_rgb, kp2,
    matches[:50],
    None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# (d) Homography using ORB matches

if len(matches) < 4:
    raise ValueError("Not enough matches to compute homography.")

src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

H_orb, mask_orb = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
warped_orb = cv.warpPerspective(img1, H_orb, (img2.shape[1], img2.shape[0]))

diff_orb = cv.absdiff(warped_orb, img2)
diff_orb_gray = cv.cvtColor(diff_orb, cv.COLOR_BGR2GRAY)


# Display results

plt.figure(figsize=(8, 6))
plt.imshow(cv.cvtColor(warped_manual, cv.COLOR_BGR2RGB))
plt.title("(a) Warped Image using Manual Points")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(diff_manual_gray, cmap="gray")
plt.title("(b) Difference Image using Manual Homography")
plt.axis("off")
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(match_img)
plt.title("(c) ORB Feature Matches")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(cv.cvtColor(warped_orb, cv.COLOR_BGR2RGB))
plt.title("(d) Warped Image using ORB + RANSAC")
plt.axis("off")
plt.show()

plt.figure(figsize=(8, 6))
plt.imshow(diff_orb_gray, cmap="gray")
plt.title("(e) Difference Image using ORB + RANSAC")
plt.axis("off")
plt.show()


# Print results

print("Clicked Points - Image 1:", pts1)
print("Clicked Points - Image 2:", pts2)
print("\nManual Homography Matrix:")
print(H_manual)
print("\nORB Homography Matrix:")
print(H_orb)
print("\nNumber of ORB keypoints in image 1:", len(kp1))
print("Number of ORB keypoints in image 2:", len(kp2))
print("Number of matches:", len(matches))
print("Number of RANSAC inliers:", int(mask_orb.sum()))
