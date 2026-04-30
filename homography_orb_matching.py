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


#(a) Manual corresponding points and homography

pts1 = np.float32([
    [159, 78],
    [513, 51],
    [553, 888],
    [93, 902],
    [317, 421],
    [385, 226]
])
pts2 = np.float32([
    [170, 98],
    [528, 78],
    [582, 886],
    [104, 921],
    [332, 435],
    [398, 244]
])

# Compute homography from image 1 to image 2
H_manual, mask_manual = cv.findHomography(pts1, pts2)

# Warp image 1 to match image 2
warped_manual = cv.warpPerspective(img1, H_manual, (img2.shape[1], img2.shape[0]))


#(b) Difference image for manual homography

diff_manual = cv.absdiff(warped_manual, img2)
diff_manual_gray = cv.cvtColor(diff_manual, cv.COLOR_BGR2GRAY)


#(c) ORB keypoints, descriptors, and matching

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

#Print results
print("Manual Homography Matrix:")
print(H_manual)
print("\nORB Homography Matrix:")
print(H_orb)
print("\nNumber of ORB keypoints in image 1:", len(kp1))
print("Number of ORB keypoints in image 2:", len(kp2))
print("Number of matches:", len(matches))
print("Number of RANSAC inliers:", int(mask_orb.sum()))
