import cv2
import os
import numpy as np
from PIL import Image
import imageio
from tqdm import trange

from myCVHelper.my_cv_helper import logger
from myCVHelper import my_cv_helper as helper


# 获取图像外围的空白区域遮罩
def get_block(im: np.ndarray) -> np.ndarray:
    mask = np.zeros(im.shape, dtype=np.int)
    # 开始BFS
    queue = []
    # 记录需要的颜色
    color = im[0][0]
    for i in range(0, im.shape[0]):
        queue.append((i, 0))
        queue.append((i, im.shape[1] - 1))
        mask[i][0] = 255
        mask[i][im.shape[1] - 1] = 255

    for i in range(0, im.shape[1]):
        queue.append((0, i))
        queue.append((im.shape[0] - 1, i))
        mask[0][i] = 255
        mask[im.shape[0] - 1][i] = 255

    directions = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1)
    ]
    # 容差
    tolerance = 3
    while not len(queue) == 0:
        top = queue[0]
        # logger.debug('get top: %s from %s' % (str(top), len(queue)))
        queue = queue[1:]
        for d in directions:
            m = (top[0] + d[0], top[1] + d[1])
            if not (0 <= m[0] < im.shape[0] and 0 <= m[1] < im.shape[1]):
                continue
            if mask[m[0]][m[1]] > 250:
                continue
            if color - tolerance <= im[m[0]][m[1]] <= color + tolerance:
                # logger.info('append: %s' % str(m))
                mask[m[0]][m[1]] = 255
                queue.append(m)

    # 最后反色
    mask = 255 - mask
    img = Image.fromarray(mask, "RGBA")
    mask = np.array(img)
    mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2GRAY)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    return mask


def process(frame: np.ndarray, filename: str) -> np.ndarray:
    src = frame
    helper.Show.imshow("src", src)
    # 初步处理图像
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    helper.Show.imshow("thresh", thresh)
    # 对二值化的图像先模糊再腐蚀才交给mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    erode = cv2.erode(thresh, kernel)
    blur = cv2.medianBlur(erode, 15)
    helper.Show.imshow("blur", blur, 1)
    helper.Show.imshow("erode", erode, 1)
    _, thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    mask = get_block(thresh2)
    # mask = get_block(erode)
    helper.Show.imshow("mask", mask, 1)
    # 扩展边界（膨胀）
    mask2 = cv2.dilate(mask, kernel)
    helper.Show.imshow("mas2", mask2, 1)
    dst = cv2.bitwise_and(src, src, mask=mask2)
    helper.Show.imshow("dst", dst, 2)
    img = Image.fromarray(dst)
    img.save(filename)
    return dst


def main():
    if not os.path.exists('./results/'):
        os.mkdir('results')
    buff = []
    # 先统计有多少图像
    total = 0
    gif = cv2.VideoCapture('Pani_poni_dash.gif')
    while True:
        ret, frame = gif.read()
        if not ret:
            break
        total += 1
    gif.release()
    gif = cv2.VideoCapture('Pani_poni_dash.gif')
    for cnt in trange(total):
        ret, frame = gif.read()
        if not ret:
            break
        filename = os.path.join("./results/", "%03d.png" % cnt)
        if os.path.exists(filename):
            buff.append(cv2.imread(filename).copy())
            continue
        buff.append(process(frame, filename).copy())
        # helper.Controls.wait_exit(0)
        helper.Controls.wait_exit(1)
    # 重复5次
    buff_temp = []
    for i in range(5):
        buff_temp.extend(buff)
    buff = buff_temp
    imageio.mimsave('pani_poni.gif', buff, 'GIF', duration=0.1)
    logger.info('done')
    os.remove('pani_poni_zipped.gif')
    os.system("ffmpeg -i pani_poni.gif pani_poni_zipped.gif")
    while True:
        # gif_dst = cv2.VideoCapture('Pani_poni_dash.gif')
        gif_dst = cv2.VideoCapture('pani_poni.gif')
        logger.info('start playing gif')
        while True:
            ret, frame = gif_dst.read()
            if not ret:
                break
            cv2.imshow("gif_dit", frame)
            logger.debug('posted one frame')
            helper.Controls.wait_exit(100)
            # helper.Controls.wait_exit(1)
        gif_dst.release()


helper.Show.resize = 1

if __name__ == '__main__':
    main()
