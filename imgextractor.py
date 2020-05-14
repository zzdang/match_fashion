import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from pathlib import Path
import os

# from types import Iterable
# __all__ = ['ImgExtractor']


class Frame:
    def __init__(self, id, frame, value):
        self.id = id
        self.frame = frame
        self.value = value

    def __lt__(self, other):
        # if not type(other) == type(self):
        # 	raise TypeError(' type of other {} must be {}'.format(other, self.__class__))
        if self.id == other.id:
            return self.id < other.id
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        # if not type(other) == type(self):
        # 	raise TypeError(' type of other {} must be {}'.format(other, self.__class__))
        return self.id == other.id 

    def __ne__(self, other):
        return not self.__eq__(other)


class ImgExtractor:
    def __init__(self, len_window=5):
        self.len_window = len_window

    def smooth(self, x, window_len, window='hanning'):
        if x.ndim != 1:
            raise ValueError("smooth only accepts 1 dimension arrays.")

        if x.size < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[2 * x[0] - x[window_len:1:-1],
                  x, 2 * x[-1] - x[-1:-window_len:-1]]
        #print(len(s))

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = getattr(np, window)(window_len)
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len - 1:-window_len + 1]

    def _vread(self, video_path):
        if not Path(video_path).exists():
            raise FileNotFoundError("not such video file found: ".format(video_path))

        cap = cv2.VideoCapture(video_path)


        curr_frame = None
        prev_frame = None

        frame_diffs = []
        frames = []
        ret, frame = cap.read()
        img_sz = frame.shape
        i = 1

        try:
            while(ret):
                luv = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
                curr_frame = luv
                if curr_frame is not None and prev_frame is not None:
                    #logic here
                    diff = cv2.absdiff(curr_frame, prev_frame)
                    count = np.sum(diff)
                    frame_diffs.append(count)
                    frame = Frame(i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), count)
                    frames.append(frame)
                prev_frame = curr_frame
                i = i + 1
                ret, frame = cap.read()
            """
                cv2.imshow('frame',luv)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            """
        finally:
            cap.release()

        return frames, frame_diffs, img_sz

    def draw_frames(self, frame_list, dir_path='.'):
        for i, frame in enumerate(frame_list):
            name = "frame_{:02d}.jpg".format(i)
            cv2.imwrite(str(Path(dir_path).joinpath(name)), frame)
            # print(name)

    def draw_diff_curve(self, diff_array, dir_path='.'):
        plt.figure(figsize=(40, 20))
        plt.locator_params(numticks=100)
        plt.stem(diff_array)
        plt.savefig(str(Path(dir_path).joinpath('plot.png')))

    def extract(self, video_path, is_save=False, dir_path="", video_name=""):# -> Iterable[np.ndarray]:

        frames, frame_diffs, img_sz = self._vread(video_path)

        diff_array = np.array(frame_diffs)
        sm_diff_array = self.smooth(diff_array, self.len_window)
        frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]

        frame_list = np.array([frames[i].frame[:,:,::-1] for i in frame_indexes])
        name_list = ["frame_{:02d}.jpg".format(i) for i in range(len(frame_indexes))]

        if is_save:
            save_path = os.path.join(dir_path,  "res", "video_extractor", video_name)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            self.draw_frames(frame_list, dir_path=save_path)
            # self.draw_diff_curve(sm_diff_array, dir_path='./i')

        return frame_list, name_list, img_sz


if __name__ == '__main__':

    extractor = ImgExtractor(len_window=10)

    extractor.extract('/media/chen/6f586f18-792a-40fd-ada6-59702fb5dabc/data/compe/demo0/video/019463.mp4', is_save=True, dir_path='./tmp/')


