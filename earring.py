import cv2 as cv

img = cv.imread(r"C:\Users\Administrator\Desktop\UOM\Sem3\Computer Vision - Dr Ranga\Assignment 02\earrings.jpg")

if img is None:
    raise FileNotFoundError("Image not found. Check the file path.")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

_, thresh = cv.threshold(gray, 240, 255, cv.THRESH_BINARY_INV)
contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

large_contours = [c for c in contours if cv.contourArea(c) > 1000]

for i, c in enumerate(large_contours):
    x, y, w, h = cv.boundingRect(c)

    pixel_size_mm = 0.0022  # sensor pixel pitch in mm
    Z = 720                 # object distance in mm
    f = 8                   # focal length in mm
    scale = (pixel_size_mm * Z) / f

    real_w = w * scale
    real_h = h * scale

    print(f"Earring {i+1}: {w} px x {h} px")
    print(f"Real size: {real_w:.2f} mm x {real_h:.2f} mm")

    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv.imwrite("earing_dimensions.png", img)
cv.imshow("earing dimensions", img)
cv.waitKey(0)
cv.destroyAllWindows()
