import os
import cv2
import numpy as np
import logging
from colorlog import ColoredFormatter
import pinyin
import timeit


def get_logger(name=__name__):
    logger_base = logging.getLogger(name)
    logger_base.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler()
    color_formatter = ColoredFormatter('%(log_color)s[%(module)-12s][%(funcName)-12s][%(levelname)-8s] %(message)s')
    # formatter = logging.Formatter('[%(module)-15s][%(funcName)-7s][%(levelname)-8s] %(message)s')
    stream_handler.setFormatter(color_formatter)
    logger_base.addHandler(stream_handler)
    return logger_base


logger = get_logger()


class HelperConfig:
    wait_time = 500
    # 最多等待一分钟就自动关闭
    wait_time_forever = 60000


class Show:
    class LineItem:
        def __init__(self, row: int = 0, height: int = 0, x: int = 0):
            self.row, self.height, self.x = row, height, x

        def __str__(self):
            return "Line[row=%s, height=%s, x=%s]" % (self.row, self.height, self.x)

    class RectItem:
        def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
            self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

        def to_tuple(self) -> tuple:
            return self.x1, self.y1, self.x2, self.y2

    offset = (300, 150)
    resize = 4
    starts = []
    windows = []

    fps_last_calls = []
    # 每n秒计算fps
    fps_calc_time = 1
    fps_last_log = 0

    @staticmethod
    def fps_get():
        time_now = timeit.default_timer()
        Show.fps_last_calls.append(time_now)
        # print(Show.fps_last_calls)
        if Show.fps_calc_time != 0:
            if len(Show.fps_last_calls) > 0 and Show.fps_last_calls[-1] - Show.fps_last_calls[0] < Show.fps_calc_time:
                return 0
        if len(Show.fps_last_calls) >= 3:
            Show.fps_last_calls = Show.fps_last_calls[1:]
        # 此时len == calc_len
        d = [1 / (Show.fps_last_calls[i] - Show.fps_last_calls[i-1])
             if (Show.fps_last_calls[i] - Show.fps_last_calls[i-1]) != 0 else 0
             for i in range(1, len(Show.fps_last_calls))]
        # print(Show.fps_last_calls, d)
        return sum(d) / len(d)

    @staticmethod
    def fps_log():
        time_now = timeit.default_timer()
        fps = Show.fps_get()
        if time_now - Show.fps_last_log < Show.fps_calc_time:
            return
        Show.fps_last_log = time_now
        logger.info('fps: %s' % str(fps))

    @staticmethod
    def window_insert(row: int, width: int, height: int):
        found = False
        y = 0
        for i in range(len(Show.starts)):
            if Show.starts[i].row == row:
                found = True
                Show.starts[i].height = max(Show.starts[i].height, height)
                Show.starts[i].x += width
                y = i
                break
        if not found:
            Show.starts.append(Show.LineItem(row=row, height=height, x=width))
            Show.starts.sort(key=lambda x: x.row)
            y = len(Show.starts) - 1
        y_sum = sum([Show.starts[i].height for i in range(y)])
        return Show.RectItem(Show.starts[y].x - width, y_sum,
                             Show.starts[y].x, y_sum + height)

    @staticmethod
    def window_clear():
        cv2.destroyAllWindows()
        Show.starts = []
        Show.windows = []

    @staticmethod
    def imshow(window_name: str, im: np.ndarray, row: int = 0, use_pinyin=True):
        if use_pinyin:
            window_name = pinyin.get(window_name, format='strip', delimiter='')
        else:
            window_name = window_name.encode('gbk').decode('utf8', errors='ignore')
        # logger.debug('im.shape: %s' % str(im.shape))
        im = cv2.resize(im, (im.shape[1] // Show.resize, im.shape[0] // Show.resize))
        im_size = im.shape
        rect = Show.window_insert(row, im_size[1], im_size[0])
        cv2.imshow(window_name, im)
        if window_name not in Show.windows:
            cv2.moveWindow(window_name, rect.x1 + Show.offset[0], rect.y1 + Show.offset[1])
            Show.windows.append(window_name)
        # logger.debug('starts now: %s' % [str(i) for i in Show.starts])


class Controls:
    @staticmethod
    def wait_exit(wait_time: int = HelperConfig.wait_time, key: int = 27):
        wait_time = HelperConfig.wait_time_forever if wait_time == 0 else wait_time
        if cv2.waitKey(wait_time) == key:
            Controls.exit_kill()

    # 干掉最后一个python进程
    @staticmethod
    def exit_kill():
        result = os.popen('wmic process where name="python.exe" list brief')
        data = result.read().split('\n')
        pid = [d.split() for d in data if len(d) != 0 and data.index(d) != 0][-1][3]
        logger.warning('Try to kill PID:%s...' % pid)
        result = os.popen("taskkill.exe /f /pid %s" % pid)
        logger.critical('result: %s' % result.read())

    # 调参
    @staticmethod
    def adjust(bar_name: str, window_name: str, im: np.ndarray, val: int, count, my_onchange, use_pinyin=True):
        if use_pinyin:
            window_name = pinyin.get(window_name, format='strip', delimiter='')
            bar_name = pinyin.get(bar_name, format='strip', delimiter='')
        else:
            window_name = window_name.encode('gbk').decode('utf8', errors='ignore')
            bar_name = bar_name.encode('gbk').decode('utf8', errors='ignore')

        def onchange(obj):
            my_onchange(window_name, im, cv2.getTrackbarPos(bar_name, window_name))

        cv2.namedWindow(window_name)
        cv2.createTrackbar(bar_name, window_name, val, count, onchange)
        # 先调用一波
        onchange(None)

    class ArgBase:
        def __init__(self, bar_name: str, val: int = 0, val_max: int = 255):
            self.bar_name, self.val, self.val_max = bar_name, val, val_max
            self.window_name, self.image = None, None

        def append_arg_window_name(self, window_name: str):
            self.window_name = window_name

    class Arg:
        def __init__(self, bar_name: str, onchange, val: int = 0, val_max: int = 255):
            self.bar_name, self.onchange, self.val, self.val_max = bar_name, onchange, val, val_max
            self.window_name, self.image = None, None

        def append_args(self, window_name: str, image: np.ndarray):
            self.window_name, self.image = window_name, image

        def call(self, obj=None):
            if self.window_name is None or self.image is None:
                logger.warning('Calling an item before appending args!')
                return
            self.onchange(self.window_name, self.image, cv2.getTrackbarPos(self.bar_name, self.window_name))

    # 调参：多个参数
    @staticmethod
    def adjust_multi(window_name: str, im: np.ndarray, args: list = None, use_pinyin=True):
        try:
            if args is None or len(args) == 0:
                return
        except TypeError:
            return
        if use_pinyin:
            window_name = pinyin.get(window_name, format='strip', delimiter='')
        else:
            window_name = window_name.encode('gbk').decode('utf8', errors='ignore')
        cv2.namedWindow(window_name)
        for i in range(len(args)):
            if use_pinyin:
                args[i].bar_name = pinyin.get(args[i].bar_name, format='strip', delimiter='')
            else:
                args[i].bar_name = bar_name = args[i].bar_name.encode('gbk').decode('utf8', errors='ignore')
            args[i].append_args(window_name, im)
            logger.debug('creating track: %s, %s' % (args[i].window_name, args[i].bar_name))
            cv2.createTrackbar(args[i].bar_name, args[i].window_name, args[i].val, args[i].val_max, args[i].call)
            # 先调用一波
            args[i].call()

    # 调参：多个参数，同时调整
    @staticmethod
    def adjust_x(window_name: str, im: np.ndarray, onchange, args: list = None, use_pinyin=True):
        try:
            if args is None or len(args) == 0:
                return
        except TypeError:
            return

        def _onchange(obj):
            onchange(window_name, im, args)

        if use_pinyin:
            window_name = pinyin.get(window_name, format='strip', delimiter='')
        else:
            window_name = window_name.encode('gbk').decode('utf8', errors='ignore')
        cv2.namedWindow(window_name)
        for i in range(len(args)):
            if use_pinyin:
                args[i].bar_name = pinyin.get(args[i].bar_name, format='strip', delimiter='')
            else:
                args[i].bar_name = bar_name = args[i].bar_name.encode('gbk').decode('utf8', errors='ignore')
            args[i].append_arg_window_name(window_name)
            logger.debug('creating track: %s, %s' % (args[i].window_name, args[i].bar_name))
            cv2.createTrackbar(args[i].bar_name, args[i].window_name, args[i].val, args[i].val_max, _onchange)
        # 先调用一波
        onchange(window_name, im, args)


class Utils:
    @staticmethod
    def extend_line(x1, y1, x2, y2, x, y, flag=1, k_=None):
        if flag == 1:
            if y1 == y2:
                return 0, y1, x, y2
            else:
                k = ((y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 1000) if k_ is None else k_
                b = (x1 * y2 - x2 * y1) / (x1 - x2) if x2 - x1 != 0 else 1000
                x3 = 0
                y3 = b
                x4 = x
                y4 = int(k * x4 + b)
            return x3, y3, x4, y4
        else:
            if x1 == x2:
                return x1, 0, x2, y
            else:
                k = ((y2 - y1) / (x2 - x1) if x2 - x1 != 0 else 1000) if k_ is None else k_
                b = (x1 * y2 - x2 * y1) / (x1 - x2) if x2 - x1 != 0 else 1000
                y3 = 0
                x3 = int(-1 * b / k)
                y4 = y
                x4 = int((y4 - b) / k)
                return x3, y3, x4, y4
