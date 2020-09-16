# function used in injection procedure
import numpy as np
import cv2


def matchByDist(lastHand, curHand):
    last = lastHand.copy()
    cur = curHand.copy()

    # 本次坐标数量比上次坐标多
    # 从上次开始遍历
    if len(last) <= len(cur):
        return match(last, cur)

    # 本次数量比上次少
    # 从本次开始遍历
    else:
        return match(cur, last)


def match(last, cur):
    totalDist = 0
    for corLast in last:
        minDist = 1000
        minCor = None
        for corCur in cur:
            dist = 0
            for index in range(4):
                dist += abs(corLast[index] - corCur[index])
            if dist < minDist:
                minDist = dist
                minCor = corCur
        totalDist += minDist
        if minCor is not None:
            cur.remove(minCor)
    return totalDist


def throwFarAwayCor(earArr, handArr):
    minIou = 1
    minIndex = 0

    if len(handArr) <= 3:
        return handArr

    for index in range(len(handArr)):
        iou = iouBetweenEarAndHand(earArr, handArr[index])
        if iou < minIou:
            minIou = iou
            minIndex = index
    return handArr.remove(handArr[minIndex])


def iouBetweenEarAndHand(earArr, handarr):
    xA = max(earArr[0], handarr[0])
    yA = max(earArr[1], handarr[1])
    xB = min(earArr[2], handarr[2])
    yB = min(earArr[3], handarr[3])

    interArea = (xB - xA + 1) * (yB - yA + 1)

    # 并集
    boxAArea = (earArr[2] - earArr[0] + 1) * (earArr[3] - earArr[1] + 1)
    boxBArea = (handarr[2] - handarr[0] + 1) * (handarr[3] - handarr[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou


def GetLargestIoUCor(earArr, handArr):
    maxIou = 0
    maxIndex = 0

    for index in range(len(handArr)):
        iou = iouBetweenEarAndHand(earArr, handArr[index])
        if iou > maxIou:
            maxIou = iou
            maxIndex = index

    xA = max(earArr[0], handArr[maxIndex][0])
    yA = max(earArr[1], handArr[maxIndex][1])
    xB = min(earArr[2], handArr[maxIndex][2])
    yB = min(earArr[3], handArr[maxIndex][3])

    return xA, yA, xB, yB


class InjectionUtil:

    def __init__(self, okDist=100, maxCache=5, zoomThreshold=50):
        self.okDist = okDist
        self.maxCache = maxCache
        self.zoomThreshold = zoomThreshold
        self.cache = Cache(maxCache=maxCache)

    def hand_stable(self, check_objects):
        earShow = check_objects[4][0]
        earArrRaw = check_objects[4][2]
        if earArrRaw.shape[0] > 0:
            earArr = earArrRaw[0]
        else:
            earArr = earArrRaw

        handShow = check_objects[3][0]
        handArrRaw = list(check_objects[3][2])

        handArr = []
        for array in handArrRaw:
            handArr.append(list(array))

        if earShow != 1:
            return 0
        throwFarAwayCor(earArr, handArr)

        if handShow != 1:
            return 0
        if self.cache.getLen() == 0:
            self.cache.addCache(handArr)
            return 0

        curDist = self.calc_dist(handArr)
        totalDist = self.cache.getTotalDist(curDist)
        self.cache.addDist(curDist)
        self.cache.addCache(handArr)

        if self.cache.getLen() < self.maxCache:
            return 0

        if totalDist <= self.okDist:
            # print('STABLE:   ', totalDist)
            return 1
        else:
            # print('UNSTABLE: ', totalDist)
            return 0

    def calc_dist(self, handArr):
        lastHand = self.cache.getLastCache()
        return matchByDist(lastHand, handArr)

    def needle_search(self, img_cv, check_objects):
        earShow = check_objects[4][0]
        earArrRaw = check_objects[4][2]
        if earArrRaw.shape[0] > 0:
            earArr = earArrRaw[0]
        else:
            earArr = earArrRaw

        handShow = check_objects[3][0]
        handArrRaw = list(check_objects[3][2])

        handArr = []
        for array in handArrRaw:
            handArr.append(list(array))

        if earShow != 1:
            return 0
        throwFarAwayCor(earArr, handArr)

        if handShow != 1:
            return 0

        xA, yA, xB, yB = GetLargestIoUCor(earArr, handArr)
        xA -= self.zoomThreshold
        yA -= self.zoomThreshold
        xB += self.zoomThreshold
        yB += self.zoomThreshold

        croppedCv = img_cv[yA: yB, xA: xB]
        if croppedCv.shape[0] <= 0 or croppedCv.shape[1] <= 1 or croppedCv.shape[2] <= 2:
            return 0
        # cv2.imwrite('cv.jpg', croppedCv)
        prop = 3
        croppedCv = cv2.resize(croppedCv,
                               (int(croppedCv.shape[0] * prop),
                                int(croppedCv.shape[1] * prop)),
                               interpolation=cv2.INTER_CUBIC)

        # orb =  cv2.ORB_create()
        # kp = orb.detect(croppedCv, None)
        # kp, des = orb.compute(img, kp)
        # croppedCv = cv2.drawKeypoints(croppedCv, kp, croppedCv)

        cv2.namedWindow('cropped cv', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('cropped cv', croppedCv)
        # cv2.destroyWindow('cropped cv')
        # cv2.imwrite('croppedCv.jpg', croppedCv)


class Cache:

    def __init__(self, maxCache=10):
        self.maxCache = maxCache

        self.cache = []
        self.cur = 0
        self.dist = []

    def getTotalDist(self, curDist):
        if self.getLen() == 0:
            return False
        if len(self.dist) == 0:
            self.dist.append(0)
            self.dist.append(curDist)
            return curDist

        totalDist = sum(self.dist) + curDist
        self.addDist(curDist)
        return totalDist

    def addDist(self, curDist):
        if len(self.dist) < self.maxCache:
            self.dist.append(curDist)
        else:
            self.dist[self.cur] = curDist

    def getLastCache(self):
        return self.cache[self.cur - 1]

    def addCache(self, item):
        if self.getLen() < self.maxCache:
            self.cache.append(item)
            self.next_cur()
        else:
            self.cache[self.cur] = item
            self.next_cur()

    def getLen(self):
        return len(self.cache)

    def next_cur(self):
        if self.cur >= self.maxCache - 1:
            self.cur = 0
        else:
            self.cur += 1


if __name__ == '__main__':
    import cv2
    from utils.Detector import Detector

    video_path = '../../video/RabbitVideo_Trim.mp4'

    jason_file_path = '../../ScoresLine.json'
    deploy_path = '../../detecting_files/no_bn.prototxt'
    model_path = '../../detecting_files/no_bn.caffemodel'

    injectionUtils = InjectionUtil()
    detector = Detector(deploy_path, model_path, jason_file_path)
    video_cap = cv2.VideoCapture(video_path)

    while video_cap.isOpened():
        ret, img = video_cap.read()
        assert (img is not None), {print('FINISHED'), exit(0)}
        imgCv = img.copy()
        # imgCv = cv2.resize(img, (960, 540))
        img = detector.check_img(img)

        checkObjects = detector.checkedObjects
        injectionUtils.needle_search(imgCv, checkObjects)

        cv2.imshow('imgDetected', img)

        keyBoard = cv2.waitKey(20)
        if keyBoard == ord('q'):
            cv2.destroyAllWindows()
            break
