# -*- coding: utf-8 -*-
#
# 检测视频
# Author: alex
# Created Time: 2019年04月30日 星期二 22时00分25秒
import cv2
import colorsys
from core.detectors import CornerNet_Saccade
from core.vis_utils import draw_bboxes

# 生成颜色列表
hsv_tuples = [(x / 80, 1., 1.) for x in range(80)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(
    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))


def detect_video(video_path, output_path, start=0, end=0, classes=None,
                 forbid_box=None):
    detector = CornerNet_Saccade()
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(output_path)
    out_fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, out_fourcc, video_fps, video_size)

    if classes is not None:
        classes = classes.split(',')

    start *= 1000
    end = end if end == 0 else end*1000
    width, height = 0, 0
    while True:
        return_value, frame = vid.read()
        if return_value is False:
            break

        msec = int(vid.get(cv2.CAP_PROP_POS_MSEC))
        if msec < start:
            continue
        if end > 0 and msec > end:
            break

        bboxes = detector(frame)
        print('当前时间进度：%.2f秒' % (msec/1000))
        image = draw_bboxes(frame, bboxes, classes=classes, colors=colors)
        out.write(image)

    print("width: %d, height: %d" % (width, height))
    out.release()


if __name__ == '__main__':
    import fire
    fire.Fire(detect_video)
