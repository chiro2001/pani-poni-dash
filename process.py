import cv2
from PIL import Image
gif = cv2.VideoCapture('Pani_poni_dash.gif')
# ret=True if it finds a frame else False. Since your gif contains only one frame, the next read() will give you ret=False
while True:
    ret, frame = gif.read()
    if not ret:
        break
    # img = Image.fromarray(frame)
    # img = img.convert('RGB')
    # img.show()
    cv2.imshow("src", frame)
    cv2.waitKey(0)
