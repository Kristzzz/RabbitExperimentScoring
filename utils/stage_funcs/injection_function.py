# function used in injection procedure
import numpy as np


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


class InjectionUtil:

    def __init__(self, okDist=100, maxCache=5):
        self.okDist = okDist
        self.maxCache = maxCache
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
    x = [[1, 2, 3]]
    print(sum(sum(x, [])))

