import cv2
import numpy as np

for i in range(1, 21):
    path = """C:/Users/32885/Desktop/true_traffic_sign/No_Parking/  (%s).jpg""" % (i)
    img = cv2.imread(path)
    for j in range(1, 21):
        new_path = """C:/Users/32885/Desktop/true_traffic_sign/No_Parking/%s.jpg""" %((i-1)*20+j)
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)

        if j < 6 or 15 < j < 21:  # resize
            cv2.imwrite(new_path, cv2.resize(img, (32, 32)))

        elif j in[6, 8]:  # 左回りに 90 度回転
            M = cv2.getRotationMatrix2D(center, 10, 1.0)
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif j in [7, 9]:  # 右回りに 90 度回転
            M = cv2.getRotationMatrix2D(center, -10, 1.0)
            res = cv2.warpAffine(img, M, (w, h))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif 9 < j <= 11:  # 明るくになる
            res = np.uint8(np.clip((cv2.add(1.5 * img, 30)), 0, 255))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif 11 < j <= 13:  # 暗くになる
            res = np.uint8(np.clip((cv2.add(0.6 * img, 0)), 0, 255))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))

        elif 13 < j <= 15:  # ぼやけた
            res = cv2.blur(img, (5, 5))
            cv2.imwrite(new_path, cv2.resize(res, (32, 32)))