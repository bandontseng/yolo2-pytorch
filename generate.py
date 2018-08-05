import os
import cv2
import numpy as np
from torch.multiprocessing import Pool

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
import sys

if len(sys.argv) != 3:
    print(f"Usage {sys.argv[0]} inputDir outputDir")


# This prevents deadlocks in the data loader, caused by
# some incompatibility between pytorch and cv2 multiprocessing.
# See https://github.com/pytorch/pytorch/issues/1355.
cv2.setNumThreads(0)


def preprocess(fname):
    # return fname
    image = cv2.imread(fname)
    im_data = np.expand_dims(
        yolo_utils.preprocess_test((image, None, cfg.multi_scale_inp_size), 0)[0], 0)
    return image, im_data


# hyper-parameters
trained_model = cfg.trained_model
thresh = 0.5
im_path = sys.argv[1]

net = Darknet19()
net_utils.load_net(trained_model, net)
net.cuda()
net.eval()
print('load model succ...')

t_det = Timer()
t_total = Timer()
im_fnames = sorted((fname
                    for fname in os.listdir(im_path)
                    if os.path.splitext(fname)[-1] == '.jpg'))
im_fnames_list = im_fnames
im_fnames = (os.path.join(im_path, fname) for fname in im_fnames)
pool = Pool(processes=1)

for i, (image, im_data) in enumerate(pool.imap(
        preprocess, im_fnames, chunksize=1)):
    t_total.tic()
    im_data = net_utils.np_to_variable(
        im_data, is_cuda=True, volatile=True).permute(0, 3, 1, 2)
    t_det.tic()
    bbox_pred, iou_pred, prob_pred = net(im_data)
    det_time = t_det.toc()
    # to numpy
    bbox_pred = bbox_pred.data.cpu().numpy()
    iou_pred = iou_pred.data.cpu().numpy()
    prob_pred = prob_pred.data.cpu().numpy()

    bboxes, scores, cls_inds = yolo_utils.postprocess(
        bbox_pred, iou_pred, prob_pred, image.shape, cfg, thresh)

    if len(cls_inds) == 0:
        print(f"Error in {im_fnames_list[i]}: I cannot find anything!")
    elif len(cls_inds) > 1:
        print(f"Error in {im_fnames_list[i]}: I find more than one object!")
    elif cls_inds[0] != 14:  # 14 is person
        print(f"Error in {im_fnames_list[i]}: I find a object({cls_inds[0]}) rather than a person!")
    else:  # normal case: only one person be found.
        x1, y1, x2, y2 = bboxes[0]
        width = x2 - x1
        height = y2 - y1

        # Because yolo draw bbox in body and head only.
        # Increase the region about 30% in top(for some individuals rasing
        # hands).
        # Increase the region about 10% in the bottom(for shoes).
        if y1 - int(height / 10) * 3 >= 0:
            y1 = int(height / 10) * 3
        else:
            y1 = 0

        if y2 + int(height / 10) < image.shape[1]:
            y2 = y2 + int(height / 10)
        else:
            y2 = image.shape[1] - 1

        newHeight = y2 - y1

        centerPoint = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

        scale = newHeight / 200  # 200 is a magic number in pose-hg-demo

        # print(f"{im_fnames_list[i]} {scale} {centerPoint} {bboxes[0]}->[{x1} {y1} {x2} {y2}]")
        with open(os.path.join(sys.argv[2], im_fnames_list[i][:-3] + 'csv'), "w") as fObj:
            fObj.write(f"{scale},{centerPoint[0]},{centerPoint[1]}\n")
