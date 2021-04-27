import argparse
import time
from pathlib import Path
import glob
import os
import pickle
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sklearn.model_selection as sms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from spacy.lang.fr import French
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pandas as pd


import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from pdf2image import convert_from_path, convert_from_bytes


nlp_model = 'model_block_classification.pkl'
vectorizer_model = 'vectorizer_block_classification.pkl'
selector_model = 'selector_block_classification.pkl'

def detect(source, img_size=640, conf_thres=0.25, iou_thres=0.45, device='cpu',
           weights='runs/train/exp_bloc/weights/last.pt', view_img=False, save_txt=True, save_conf=False,
           classes=None, agnostic_nms=False, augment=False, update=False, project='runs/detect', name='exp',
           exist_ok=False, save_img=False):
    """Input:The converted pdf to image, the weights
    Output:The detected Bloc on the image and also the labels of the predicted coordinates"""
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    return save_dir


def get_coordinate(label, width, height):
    """Input:The label, width and height of the image
    Output: The coordinates of the image"""
    x_c, y_c, w, h = label[0], label[1], label[2], label[3]
    
    x_min = 2 * width * x_c - (width/2)*(2*x_c + w)
    x_max = (width/2)*(2*x_c + w)
    
    y_min = 2 * height * y_c - (height/2)*(2*y_c + h)
    y_max = (height/2) * (2*y_c + h)
    
    return int(x_min), int(x_max), int(y_min), int(y_max)


def tesseract_gars_sur(img):
    """Input:The converted pdf to image
    Output: The cutting bloc of the image"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)

    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    out_below = pytesseract.image_to_string(img)

    return out_below


def bloc(labels, images):
    """Input: The convert pdf into image and the predicted labels
    Output: A dictionnary with all the predicted bloc"""
    with open(labels, 'r') as f:
        labels = f.read()
    labels = [l.split(' ') for l in labels.split("""\n""")]
    final_labels = list()
    for w in labels[:-1]:
        final_labels.append([float(x) for x in w])
    bloc_dict = {}
    cnt = 0
    for label in final_labels:
        l, coordinate = label[0], label[1:]
        img = cv2.imread(images)
        h, w, _ = img.shape
        x_min, x_max, y_min, y_max = get_coordinate(coordinate, w, h)
        bloc_dict[cnt] = tesseract_gars_sur(img[y_min:y_max, x_min:x_max])
        cnt += 1
        
    return bloc_dict

def new_predict(model, sample, vectorizer, selector):
    """Input: model (pkl file), string text, vectorizer and selector (2 methods returned by training
    for preprocessing)
       Output: prediction (block label)"""
    new_sample = pd.Series(sample)
    sample_preprocessed = vectorizer.transform(new_sample)
    sample_preprocessed = selector.transform(sample_preprocessed)
    prediction = model.predict(sample_preprocessed)
    return prediction


if __name__ == "__main__":
    cv_pdf = input('PDF :')
    pages = convert_from_path(cv_pdf, 500)
    print(pages)
    for page in pages:
        image = page.save(cv_pdf[:-4] + '.jpg', 'JPEG')

    print(glob.glob("*.jpg"))
    #image_path = glob.glob("*.jpg")[0]

    directory = detect(image)

    for file in glob.glob(str(directory)+'/labels/*.txt'):
        labels = file

    final_dict = bloc(labels, image_path)
    print(final_dict)

    nlp_model_loaded = pickle.load(open(nlp_model, 'rb'))
    vectorizer_model_loaded = pickle.load(open(vectorizer_model, 'rb'))
    selector_model_loaded = pickle.load(open(selector_model, 'rb'))

    for key in final_dict:
        prediction = new_predict(nlp_model_loaded, pd.Series(final_dict[key]), vectorizer_model_loaded, selector_model_loaded)
        print(prediction)