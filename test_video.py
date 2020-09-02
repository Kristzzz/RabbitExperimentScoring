"""test file."""
import cv2
from utils.GiveMark import GradeSYS
from utils.Detector import Detector

video_path = './video/RabbitVideo.mp4'

jason_file_path = './ScoresLine.json'
deploy_path = './detecting_files/no_bn.prototxt'
model_path = './detecting_files/no_bn.caffemodel'

last_stage = ''
num_of_stage = 3

detector = Detector(deploy_path, model_path, jason_file_path)
grade_sys = GradeSYS(jason_file_path)

video_cap = cv2.VideoCapture(video_path)

while video_cap.isOpened():
    ret, img = video_cap.read()
    img = detector.check_img(img)
    w, h, *_ = img.shape
    # detector.print_checked_objects()
    obj = detector.checkedObjects
    flag = grade_sys.begin_mark_line(detector.checkedObjects, w, h)
    if flag:
        # grade_sys.print_transcript()
        transript = grade_sys.get_transcript()
        now_stage = grade_sys.get_now_stage()

        string_1 = 'Now score is:'
        string_2 = '{}'.format(transript)
        cv2.putText(img, string_1, (20, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, string_2, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, now_stage, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow('Rabbit Experiments & Now scoring:', img)
    keyBoard = cv2.waitKey(1)
    if keyBoard == ord('q'):
        cv2.destroyAllWindows()
        break

transript = grade_sys.get_transcript()
if len(transript) < num_of_stage:
    grade_sys.set_transcript(num_of_stage, 0)

print('Now experiment ends.')
print('Your final scoring is: ')
grade_sys.print_transcript()
