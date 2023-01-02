import cv2
import numpy as np


def to_gray(image: np.ndarray):
  return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def sobel_gradient(gray_image: np.ndarray):
  blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
  # sobel gradient
  gradX = cv2.Sobel(blurred, ddepth = cv2.CV_32F, dx=1, dy=0)
  gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)

  # thresh and blur
  blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
  return cv2.threshold(blurred, thresh=100, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1], blurred

def morphology(threshed_image: np.ndarray, erode_iterations: int, dilate_iterations: int):
  H, W = threshed_image.shape
  if H > W:
    kernel_size = (int(W / 18), int(H/40))
  else:
    kernel_size = (int(W / 40), int(H / 18))
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
  
  morpho_image = cv2.morphologyEx(threshed_image, cv2.MORPH_CLOSE, kernel)
  morpho_image = cv2.erode(morpho_image, None, iterations=erode_iterations)
  morpho_image = cv2.dilate(morpho_image, None, iterations=dilate_iterations)

  return morpho_image

def crop(morpho_image: np.ndarray, source_image: np.ndarray):
  contours, _ = cv2.findContours(morpho_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
  crops = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
  croped = []
  croped_points = []
  H, W, C = source_image.shape
  for c in crops:
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    H, W, C = source_image.shape
    total = H * W
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]

    x1 = max(min(Xs), 0)
    x2 = min(max(Xs), W)
    y1 = max(min(Ys), 0)
    y2 = min(max(Ys), H)

    new_height, new_width = y2-y1, x2-x1
    if new_height < H / 4 or new_width < W / 4: ## 잘린 이미지의 가로와 세로의 길이가 일정 비율보다 작다면 그냥 crop 하지 않고 사용한다.
      break
    else:
      croped.append(source_image[y1:y1+new_height, x1:x1 + new_width])
      croped_points.append((x1, x2, y1, y2))
  if len(croped) == 0:
    croped.append(source_image)
    croped_points.append((0, W, 0, H))
  return croped, croped_points