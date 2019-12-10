
"""test file."""
import cv2
import time
from utils.GiveMark import GradeSYS
from utils.Detector import Detector

video_path = './video/RabbitVideo.mp4'

jason_file_path = './ScoresLine.json'
deploy_path = './detecting_files/no_bn.prototxt'
model_path = './detecting_files/no_bn.caffemodel'

detector = Detector(deploy_path, model_path, jason_file_path)
grade_sys = GradeSYS(jason_file_path)

video_cap = cv2.VideoCapture(video_path)

while video_cap.isOpened():
    ret, img = video_cap.read()
    img = detector.check_img(img)
    detector.print_checked_objects()
    obj = detector.checkedObjects
    grade_sys.begin_mark_line(detector.checkedObjects)
    # grade_sys.print_transcript()
    transript = grade_sys.get_transcript()


    string_1 = 'Now score is:'
    string_2 = '{}'.format(transript)
    cv2.putText(img, string_1, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, string_2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)


    cv2.imshow('Rabbit Experiments & Now scoring:',
               img)
    cv2.waitKey(25)
print('Now experiment ends.')
print('Your final scoring is: ')
grade_sys.print_transcript()
