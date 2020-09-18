import json
import sys
import numpy as np
import cv2

import utils.stage_funcs.injection_function

sys.path.append('utils/')


# 评分系统 维护一个实现了 GiveMark 接口的函数队列 用于给当前的 checkedObjects 评分
class GradeSYS:
    def __init__(self, configFile):
        self.pakage = __import__("GiveMark")  # 获取当前包的引用
        self.transcript = {}  # 维护一个成绩单字典 用于保存评分结果
        self.commandLine = []  # 维护一个评分操作的列表

        with open(configFile, "r", encoding="utf-8") as fp:  # 由 config 确定哪些评分项需要被载入
            config = json.load(fp)
            self.stageList = config['scoresLine']

        for stage in self.stageList:
            stage_func_name = getattr(self.pakage, stage)  # 获取目标类的引用
            stage_func = stage_func_name(self.transcript)  # 给入成绩单，创建一个评分对象
            self.commandLine.append(stage_func)  # 将评分对象加入执行链

    def begin_mark_line(self, checkedObjects, w, h, origin_img, detected_img):
        if len(self.commandLine) < 1:
            return False
        # 给入当前的目标检测情况 开始一个轮次的评分
        for stage_func in self.commandLine:
            if stage_func.give_mark(checkedObjects, w, h, origin_img, detected_img):
                # 如果返回值为 True 则表示该项测试已结束 将其移出评分序列
                self.commandLine.remove(stage_func)
            else:
                break
        return True

    def get_now_stage(self):
        return "{}".format(self.commandLine[0])

    def print_transcript(self):
        print(self.transcript)

    def get_transcript(self):
        return self.transcript

    def set_transcript(self, name, score):
        self.transcript[name] = score


# 所有评分函数的接口
class GiveMark:
    def __init__(self, transcript):
        """
        目前已经检测到的目标  checkedObjects 为一个列表，下标即为对应的 Object
        0：兔子  1：剪刀  2：伤口  3：手  4：耳朵  5：针头
        列表的值:
         [0]:是否出现
         [1]: Object 到目前为止出现的次数(为了减小错误识别造成的影响，暂定出现40次时确定该物品已被检测到)
         [2]:一个标识 Object 当前位置的元组
        """

        self.transcript = transcript
        self.minTimes = 40
        self.Ucount =[0,0]

    def give_mark(self, transcript, w, h, origin_img, detected_img):
        print("You got 99 points!")


# =====================所有的评分方法在此添加============================================
class CheckCatching(GiveMark):
    def give_mark(self, checkedObjects, w, h, origin_img, detected_img):
        # 针头出现 阶段判0
        if checkedObjects[5][0] == 1:
            self.transcript['Catch'] = 0
            return True

        # if len(checkedObjects[0][2]) > 0:
        #     rabbit_pos = checkedObjects[0][2][0]
        #     rabbit_center_x = (rabbit_pos[0] + rabbit_pos[2]) / 2
        #     rabbit_center_y = (rabbit_pos[1] + rabbit_pos[3]) / 2
        #
        #     if 640 < rabbit_center_x < 1280 and 360 < rabbit_center_y < 720:
        #         # print('有效的抓拿判断帧...')
        #         if checkedObjects[3][0] == 1 and checkedObjects[4][0] == 1:
        #             hand_pos = checkedObjects[3][2][0]
        #             ear_pos = checkedObjects[4][2][0]
        #             judge = self.judge_catching(hand_pos, ear_pos)
        #             if judge:
        #                 self.transcript['Catch'] = 10
        #             else:
        #                 self.transcript['Catch'] = 0
        #             # print('抓拿判定结束')
        #             return True

        if len(checkedObjects[0][2] > 0):
            rabbit_pos = checkedObjects[0][2][0]
            rabbit_c_x = (rabbit_pos[0] + rabbit_pos[2]) / 2 / w
            rabbit_c_y = (rabbit_pos[1] + rabbit_pos[3]) / 2 / h

            if 640 / w < rabbit_c_x < 1280 / w and 360 / h < rabbit_c_y < 720 / h:
                if len(checkedObjects[3][2]) < 2 and checkedObjects[3][0] == 1:  # 手出现 且 只有1个坐标数组
                    if checkedObjects[4][0] == 1:
                        hand_pos = checkedObjects[3][2][0]
                        ear_pos = checkedObjects[4][2][0]
                        judge = self.judge_catching(hand_pos, ear_pos, w, h)
                        if judge:
                            self.transcript['Catch'] = 10
                        # print('抓拿判定结束')
                        return True

    def judge_catching(self, hand_pos, ear_pos, w, h):
        for i in range(4):
            if i % 2 == 0:
                hand_pos[i] /= w
                ear_pos[i] /= w
            else:
                hand_pos[i] /= h
                ear_pos[i] /= h
        # cal iou
        iou = hand_pos.copy()
        iou[0] = max(hand_pos[0], ear_pos[0])
        iou[1] = max(hand_pos[1], ear_pos[1])
        iou[2] = min(hand_pos[2], ear_pos[2])
        iou[3] = min(hand_pos[3], ear_pos[3])

        if iou[0] >= iou[2] or iou[1] >= iou[3]:
            iou_res = 0
        else:
            s_iou = (iou[2] - iou[0]) * (iou[3] - iou[1])
            s = (hand_pos[2] - hand_pos[0]) * (hand_pos[3] - hand_pos[1])
            iou_res = s_iou / s

        if iou_res < .5:
            return True
        else:
            return False


class CheckNeedle(GiveMark):
    """
    TODO
    """
    def __init__(self, transcript):
        super().__init__(transcript)
        self.checkMax = 40
        self.injectionCheck = utils.stage_funcs.injection_function.InjectionUtil()
        self.checkList = np.zeros(self.checkMax)
        self.cur = 0

        self.stage = False

    def nextCur(self):
        self.cur += 1
        if self.cur >= self.checkMax:
            self.cur = 0

    def give_mark(self, checkedObjects, w, h, origin_img, detected_img):
        if checkedObjects[2][0] == 1:
            self.transcript['Needle'] = 0
            print('针头检测结束')
            self.stage = False
            return True

        if checkedObjects[5][0] == 1:
            self.stage = True

        # print("针头检测开始...")
        if self.stage:
            self.injectionCheck.needle_search(origin_img, checkedObjects)
            self.checkList[self.cur] = self.injectionCheck.hand_stable(checkedObjects)
            self.nextCur()

            true = 0
            for check in self.checkList:
                if check == 1:
                    true += 1
            if true == self.checkMax:
                self.transcript['Needle'] = 10
                print('针头检测结束')
                self.stage = False
                return True
            return False


class CheckFixed(GiveMark):


    def __init__(self, transcript):
        super().__init__(transcript)
        self.count = 0
        self.stage = False

    def give_mark(self, checkedObjects, w, h, origin_img, detected_img):
        if checkedObjects[2][0] == 1:
            self.transcript['fixed'] = 0
            print('固定检测结束')
            self.stage = False
            return True

        if len(checkedObjects[0]) >0 and len(checkedObjects[0][2]>0):
            rabbit_pos = checkedObjects[0][2][0]
            x_start = rabbit_pos[0]
            y_start = rabbit_pos[1]
            x_end = rabbit_pos[2]
            y_end = rabbit_pos[3]
            rabbit_center_x = int((x_start / 2 + x_end / 2) / 2)
            rabbit_center_y = int((y_start / 2 + y_end / 2) / 2)

            # print("检测开始")

            cropImg0 = detected_img[int(y_start / 2):(rabbit_center_y), int(x_start / 2):int(x_end / 2)]
            cropImg1 = detected_img[(rabbit_center_y):int(y_end / 2), int(x_start / 2):int(x_end / 2)]

            # cropImg2 = cv2.flip(cropImg1, 0)
            cropImg2 = cropImg1[:, ::-1]
            # print("翻转完成")

            H1 = cv2.calcHist([cropImg0], [1], None, [256], [0, 256])
            H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理
            # print("H1计算完成")
            # 计算图img2的直方图
            H2 = cv2.calcHist([cropImg2], [1], None, [256], [0, 256])
            H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)
            # print("H2计算完成")

            # 利用compareHist（）进行比较相似度
            similarity = cv2.compareHist(H1, H2, 0)
            print("对称度为", '%.4f' % (similarity * 100), "%")

            # img1和img2直方图展示
            # plt.subplot(2, 1, 1)
            # plt.plot(H1)
            # plt.subplot(2, 1, 2)
            # plt.plot(H2)
            # plt.show()
            if similarity > 0.9:
                self.count = self.count + 1
            elif similarity > 0.8:
                self.count = self.count + 0.5
            elif similarity > 0.7:
                self.count = self.count + 0.05
            print(self.count)
            if self.count > 200:
                print("固定成功")
                self.transcript['Fixed'] = 10
                return True
        elif checkedObjects[0][2] == 0:
            return False
    # print("固定检测结束")
        return False

class CheckWound(GiveMark):
    # 检查伤口是否存在
    def give_mark(self, checkedObjects, w, h, origin_img, detected_img):
        # print("伤口检测开始...")
        if checkedObjects[2][1] > self.minTimes:
            self.transcript['Wound'] = 10
            # print("检测到伤口")
            return True
        # print("伤口检测结束")
        return False


class CheckU(GiveMark):
    # 检测U型管摆放是否正确
    def give_mark(self, checkedObjects, w, h, origin_img, detected_img):
        if checkedObjects[2][0] == 1:
            self.Ucount[0] = 1
            wound = checkedObjects[2][2][0]
            imtem = cv2.imread('./template/tem.jpg')
            img = cv2.imread('./template/CurrentImg.jpg')
            wou = img[wound[1]:wound[3], wound[0]:wound[2]]
            # 处理模板 --------------------------------
            dstW = cv2.cvtColor(imtem, cv2.COLOR_BGR2HSV)
            low_hsv = np.array([0, 0, 46])
            high_hsv = np.array([180, 85, 255])
            dst = cv2.inRange(dstW, low_hsv, high_hsv)
            kernel = np.ones((5, 5), np.uint8)
            dilate_result = cv2.dilate(dst, kernel)
            erode_img = cv2.erode(dilate_result, kernel)
            erode_img = cv2.erode(erode_img, kernel)
            dilate_result = cv2.dilate(erode_img, kernel)
            b1 = dilate_result
            imtem = b1
            # 根据HSV提取伤口 --------------------------------
            dstW = cv2.cvtColor(wou, cv2.COLOR_BGR2HSV)
            low_hsv = np.array([0, 0, 46])
            high_hsv = np.array([180, 85, 255])
            dst = cv2.inRange(dstW, low_hsv, high_hsv)
            r, b1 = cv2.threshold(dst, 125, 255, cv2.THRESH_BINARY_INV)
            kernel = np.ones((5, 5), np.uint8)
            dilate_result = cv2.dilate(b1, kernel)
            erode_img = cv2.erode(dilate_result, kernel)
            erode_img = cv2.erode(erode_img, kernel)
            dilate_result = cv2.dilate(erode_img, kernel)
            b1 = dilate_result
            # orb匹配
            w, h = b1.shape
            w2, h2 = imtem.shape
            img1 = cv2.resize(b1, (h2, w2))
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(b1, None)
            kp2, des2 = orb.detectAndCompute(imtem, None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            img3 = cv2.drawMatches(b1, kp1, imtem, kp2, matches[:80], imtem, flags=2)
            # knn匹配
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)
            # 最大匹配点数目
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)
            if len(matches) != 0:
                similary = float(len(good)) / len(matches)
            else:
                similary = 0
            if similary > 0.33:
                print("判断为ture,相似度:%s" % similary, "u行管正确插入，积分")
                self.Ucount[1] = self.Ucount[1] + 1
        if self.Ucount[0] == 0:
            return False
        # 标志位为0时，还未检测到伤口，不进入该阶段；
        if self.Ucount[1] < 5:
            return False
        # 错误判定条件仍然需要考虑
        else:
            self.transcript['Uright'] = 10
            print("U行管检测结束")
            return True


class CheckNerve(GiveMark):
    """
    TODO
    """
    times = 0

    def give_mark(self, checkedObjects, w, h, origin_img, detected_img):
        self.times += 1
        # print("神经检测开始...")
        if self.times >= 20:
            self.transcript['Nerve'] = 10
            return True
        # print("神经检测结束")
        return False


# ======================================================================================


# 测试
if __name__ == '__main__':
    gs = GradeSYS("../ScoresLine.json")

    checkedObjects = [[] for i in range(6)]
    checkedObjects[2].append(43)
    checkedObjects[2].append((1, 1, 1, 1))
    checkedObjects.append([0, 1, 2, 3])
    print(checkedObjects)

    gs.begin_mark_line(checkedObjects)

    gs.print_transcript()
