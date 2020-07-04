import cv2
import glob
import numpy as np

from utils.GiveMark import GradeSYS
from utils.time_clock import clock


class Detector:
    """
    运行的检测类，用于检测当前图片上的物品，并维护一个 List checkedObject
    """

    def __init__(self,deploy_path, model_path, jason_file_path):
        self._deploy_path = deploy_path
        self._model_path = model_path
        self._jason_file_path = jason_file_path
        self._size = 300
        self._classNum = 6  # 目前能够识别的物品数量
        """
        列表的值:
        一项为 Object 到目前为止出现的次数(为了减小错误识别造成的影响，暂定出现40次时确定该物品已被检测到) , 第二项为一个标识 Object 当前位置的元组
        [0]当前帧是否有目标
        [1]目标出现次数
        [2]目标坐标
        """
        res = np.empty(shape=[0, 4], dtype=int)
        self.checkedObjects = [[0, 0, res] for i in range(self._classNum)]  # 当前帧是否有目标，目标出现次数，目标坐标；二维列表每一个元素代表一个对象

        self._net = cv2.dnn.readNetFromCaffe(self._deploy_path, self._model_path)
        self._clock = clock()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, new_size):
        self._size = new_size

    @property
    def checkedObjects(self):
        return self._checkedObjects

    @checkedObjects.setter
    def checkedObjects(self, new_checked_objects):
        self._checkedObjects = new_checked_objects

    def print_checked_objects(self):
        obj_name = ['兔子', '剪刀', '伤口', '手', '耳朵', '针头']
        checked_obj = self._checkedObjects[:6]
        stage_name = ['抓拿', '麻醉', '固定', '手术']
        now_stage = stage_name[self._checkedObjects[5][0]]
        print("Stage :", now_stage)
        for obj_name, obj in zip(obj_name, checked_obj):
            print("【", obj_name, "】", ": ", obj[0], " ", obj[1], obj[2], end=" ")
        print()

    def reset_obj(self):
        for obj in self._checkedObjects:
            if len(obj) > 1:
                obj[0] = 0  # 重置目标是否出现

    def set_net_input(self, img):
        self._net.setInput(cv2.dnn.blobFromImage(img,
                                                 1.0 / 127.5,
                                                 (300, 300),
                                                 (127.5, 127.5, 127.5),
                                                 swapRB=False, crop=False))

    def check_img(self, img):
        assert (isinstance(img, np.ndarray)), "输入对象不是图片!"
        self.reset_obj()

        # 1. Find and rectangle object
        img_w, img_h, *_ = img.shape
        self.set_net_input(img)
        cv_out = self._net.forward()

        # 清零每一帧的bounding box
        for i in range(self._classNum):
            self.checkedObjects[i][2] = np.empty(shape=[0, 4], dtype=int)

        for detection in cv_out[0, 0, :, :]:
            """
            Detection: shape is (1, 7), which is (response, class, score, coords).
            """
            score = float(detection[2])
            if score > 0.3:
                left = detection[3] * img_h
                top = detection[4] * img_w
                right = detection[5] * img_h
                bottom = detection[6] * img_w

                boudingbox = [int(left), int(top), int(right), int(bottom)]

                self._checkedObjects[int(detection[1] - 1)][0] = 1
                self._checkedObjects[int(detection[1] - 1)][1] += 1
                self._checkedObjects[int(detection[1] - 1)][2] = \
                    np.append(self.checkedObjects[int(detection[1] - 1)][2], [boudingbox], axis=0)

                cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
                text = str(detection[1])
                cv2.putText(img, text, (int(left), int(top)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)

        img = cv2.resize(img, (int(img_h / 2), int(img_w / 2)))
        return img


# 测试
if __name__ == '__main__':
    jason_file_path = '../ScoresLine.json'
    detector = Detector(jason_file_path)
    grade_sys = GradeSYS(jason_file_path)
    imgStream = glob.glob("../img_catching/*.png")  # 获取img路径下的所有 jpg 图片
    for im in imgStream:
        im = cv2.imread(im)
        im = detector.check_img(im)
        detector.print_checked_objects()
        grade_sys.begin_mark_line(detector.checkedObjects)
        grade_sys.print_transcript()
        cv2.imshow("img", im)
        k = cv2.waitKey(0)
        if k == ord('q'):
            break
