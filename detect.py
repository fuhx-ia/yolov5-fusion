# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative ROOT设置为相对路径 .
# 用户自定义的库，由于上一步已经把路径加载上了，所以现在可以导入，这个顺序不可以调换
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


'''===================1.载入参数======================='''
@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL 事先训练完成的权重文件
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam) 预测时的输入数据，可以是文件/路径/URL/glob, 输入是0的话调用摄像头作为输入
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path dataset.yaml文件路径，包括类别/图片/标签等信息
        imgsz=(640, 640),  # inference size (height, width)  预测时的放缩后图片大小(因为YOLO算法需要预先放缩图片), 两个值分别是height, width。
        conf_thres=0.25,  # confidence threshold 置信度阈值, 高于此值的bounding_box才会被保留。默认0.25，用在nms中
        iou_thres=0.45,  # NMS IOU threshold IOU阈值,高于此值的bounding_box才会被保留。默认0.45，用在nms中
        max_det=1000,  # maximum detections per image 一张图片上检测的最大目标数量，用在nms中
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results 是否在屏幕上显示检测结果，
        save_txt=False,  # save results to *.txt 是否将检测结果保存为文本文件
        save_conf=False,  # save confidences in --save-txt labels 是否在保存的文本文件中包含置信度信息
        save_crop=False,  # save cropped prediction boxes  是否保存裁剪后的预测框
        nosave=False,  # do not save images/videos 不保存图片、视频, 要保存图片
        classes=None,  # filter by class: --class 0, or --class 0 2 3 过滤指定类的预测结
        agnostic_nms=False,  # class-agnostic NMS 进行NMS去除不同类别之间的框
        augment=False,  # augmented inference TTA测试时增强/多尺度预测，可以提分
        visualize=False,  # visualize features 是否可视化网络层输出特征
        update=False,  # update all models 如果为True,则对所有模型进行strip_optimizer操作,去除pt文件中的优化器等信息
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment 推理结果覆盖之前的结果
        line_thickness=3,  # bounding box thickness (pixels)  绘制Bounding_box的线宽度
        hide_labels=False,  # hide labels 隐藏标签
        hide_conf=False,  # hide confidences  隐藏置信度
        half=False,  # use FP16 half-precision inference  是否使用半精度推理（节约显存）
        dnn=False,  # use OpenCV DNN for ONNX inference  是否使用OpenCV DNN预测
        vid_stride=1,  # video frame-rate stride
):
    '''=========================2.初始化配置==========================='''
    source = str(source)
    # 是否保存图片和txt文件，如果nosave(传入的参数)为false且source的结尾不是txt则保存图片
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    # 判断source是不是视频/图像文件路径
    # Path()提取文件名。suffix：最后一个组件的文件扩展名。若source是"D://YOLOv5/data/1.jpg"， 则Path(source).suffix是".jpg"， Path(source).suffix[1:]是"jpg"
    # 而IMG_FORMATS 和 VID_FORMATS两个变量保存的是所有的视频和图片的格式后缀。
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    # 判断source是否是链接
    # .lower()转化成小写 .upper()转化成大写 .title()首字符转化成大写，其余为小写, .startswith('http://')返回True or Flase
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    # 判断是source是否是摄像头
    # .isnumeric()是否是由数字组成，返回True or False
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        # 返回文件。如果source是一个指向图片/视频的链接,则下载输入数据
        source = check_file(source)  # download

    '''========================3.保存结果======================'''
    # Directories  save_dir是保存运行结果的文件夹名，是通过递增的方式来命名的。
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    '''=======================4.加载模型=========================='''
    # Load model
    # 获取设备 CPU/CUDA
    device = select_device(device)
    # DetectMultiBackend定义在models.common模块中，是我们要加载的网络，其中weights参数就是输入时指定的权重文件（比如yolov5s.pt）
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    # stride：推理时所用到的步长，默认为32， 大步长适合于大目标，小步长适合于小目标
    # names：保存推理结果名的列表，比如默认模型的值是['person', 'bicycle', 'car', ...]
    # pt: 加载的是否是pytorch模型（也就是pt格式的文件）
    stride, names, pt = model.stride, model.names, model.pt
    # 确保输入图片的尺寸imgsz能整除stride=32 如果不能则调整为能被整除并返回
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    '''=======================5.加载数据========================'''
    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 推理部分
    # 使用空白图片（零矩阵）预先用GPU跑一遍预测流程，可以加速预测
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        '''
         在dataset中，每次迭代的返回值是self.sources, img, img0, None, ''
          path：文件路径（即source）
          im: resize后的图片（经过了放缩操作）CHW [3,480,640]
          im0s: 原始图片 HWC
          vid_cap=none
          s： 图片的打印信息，比如路径，大小
        '''
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim  添加一个第0维。缺少batch这个尺寸，所以将它扩充一下，变成[1，3,640,480]

        # Inference
        with dt[1]:
            # 可视化文件路径。如果为True则保留推理过程中的特征图，保存在runs文件夹中
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            # 模型预测出来的所有检测框，torch.size=[1,18900,85]
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            '''
             pred: 网络的输出结果 [1, 18900, 85]->[1, 5, 6] 6 = x,y,x,y,conf,cls
             conf_thres： 置信度阈值
             iou_thres： iou阈值
             classes: 是否只保留特定的类别 默认为None
             agnostic_nms： 进行nms是否也去除不同类别之间的框
             max_det: 检测框结果的最大数量 默认1000
            '''
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        # 把检测框画到原图中
        for i, det in enumerate(pred):  # per image
            '''
                i：每个batch的信息
                det:表示5个检测框的信息 [5,6]
            '''
            seen += 1   # seen是一个计数的功能
            if webcam:  # batch_size >= 1
                # 如果输入源是webcam则batch_size>=1 取出dataset中的一张图片
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '   # s后面拼接一个字符串i
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # 图片/视频的保存路径save_path 如 runs\\detect\\exp8\\fire.jpg
            save_path = str(save_dir / p.name)  # im.jpg
            # 设置保存框坐标的txt文件路径，每张图片对应一个框坐标信息
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            # 设置输出图片信息。图片shape (w, h)
            s += '%gx%g ' % im.shape[2:]  # print string
            # 得到原图的宽和高[1024,768,1024,768]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            # 保存截图。如果save_crop的值为true，则将检测到的bounding_box单独保存成一张图片
            imc = im0.copy() if save_crop else im0  # for save_crop
            # 得到一个绘图的类，类中预先存储了原图、线条宽度、类名
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # 判断有没有框
            if len(det):
                # Rescale boxes from img_size to im0 size
                # 将预测信息映射到原图
                # 将标注的bounding_box大小调整为和原图一致（因为训练时原图经过了放缩）此时坐标格式为xyxy
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round() #scale_coords：坐标映射功能

                # Print results
                # 打印检测到的类别数量
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # 保存预测结果：txt/图片画框/crop-image
                for *xyxy, conf, cls in reversed(det):
                    # 将每个图片的预测信息分别存入save_dir/labels下的xxx.txt中 每行: class_id + score + xywh
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            # 写入对应的文件夹里，路径默认为“runs\detect\exp*\labels”
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下，在原图像画图或者保存结果
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class 类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))  # 绘制边框
                    # 在原图上画框+将预测到的目标剪切出来保存成图片，保存在save_dir/crops下（单独保存）
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            im0 = annotator.result()
            # 设置显示图片
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # 设置保存图片/视频
            if save_img:
                if dataset.mode == 'image':  # 如果是图片就保存
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video  说明这张图片属于一段新的视频,需要重新创建视频文件
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'runs/train/exp/weights/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'M3FD/images/test', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/m3fd.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
