# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset.
Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (LOGGER, TQDM_BAR_FORMAT, check_amp, check_dataset, check_file, check_git_info,
                           check_git_status, check_img_size, check_requirements, check_suffix, check_yaml, colorstr,
                           get_latest_run, increment_path, init_seeds, intersect_dicts, labels_to_class_weights,
                           labels_to_image_weights, methods, one_cycle, print_args, print_mutation, strip_optimizer,
                           yaml_save)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (EarlyStopping, ModelEMA, de_parallel, select_device, smart_DDP, smart_optimizer,
                               smart_resume, torch_distributed_zero_first)

# 分布式训练初始化
LOCAL_RANK = int(os.getenv('LOCAL_RANK',
                           -1))  # https://pytorch.org/docs/stable/elastic/run.html  # -本地序号。这个 Worker 是这台机器上的第几个 Worker
RANK = int(os.getenv('RANK', -1))  # -进程序号。这个 Worker 是全局第几个 Worker
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))  # 总共有几个 Worker
GIT_INFO = check_git_info()
'''
   查找名为LOCAL_RANK，RANK，WORLD_SIZE的环境变量，
   若存在则返回环境变量的值，若不存在则返回第二个参数（-1，默认None）
rank和local_rank的区别： 两者的区别在于前者用于进程间通讯，后者用于本地设备分配。
'''


def train(hyp,  # 超参数 可以是超参数配置文件的路径或超参数字典
          opt,  # main中opt参数
          device,  # 当前设备
          callbacks  # 用于存储Loggers日志记录器中的函数，方便在每个训练阶段控制日志的记录情况
          ):  # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
            opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze
    callbacks.run('on_pretrain_routine_start')

    # Directories 创建训练权重目录，设置模型保存路径
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters 读取超参数配置文件
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Save run settings 保存训练中的参数 hyp和opt
    if not evolve:
        yaml_save(save_dir / 'hyp.yaml', hyp)
        yaml_save(save_dir / 'opt.yaml', vars(opt))

    # 定义数据集字典
    data_dict = None
    # Loggers 设置wandb和tb两种日志, wandb和tensorboard都是模型信息，指标可视化工具
    if RANK in {-1, 0}:  # 如果进程编号为-1或0
        # 初始化日志记录器实例
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):  # 将日志记录器中的方法与字符串进行绑定
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config  配置:画图开关,cuda,种子,读取数据集相关的yaml文件
    # 是否绘制训练、测试图片、指标图等，使用进化算法则不绘制
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    # 设置随机种子
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    # 加载数据配置信息
    with torch_distributed_zero_first(LOCAL_RANK):  # 同步所有进程
        data_dict = data_dict or check_dataset(data)  # check if None 检查数据集，如果没找到数据集则下载数据集(仅适用于项目中自带的yaml文件数据集)
    # 获取训练集、测试集图片路径
    train_path, val_path = data_dict['train'], data_dict['val']
    # nc：数据集有多少种类别
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # names: 数据集所有类别的名字，如果设置了single_cls则为一类
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    # 当前数据集是否是coco数据集(80个类别)
    is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset

    # Model 载入模型
    # 检查后缀
    check_suffix(weights, '.pt')  # check weights
    # 加载预训练权重 yolov5提供了5个不同的预训练权重，可以根据自己的模型选择预训练权重
    pretrained = weights.endswith('.pt')
    # 预训练模型加载
    if pretrained:
        with torch_distributed_zero_first(LOCAL_RANK):  # 用于同步不同进程对数据读取的上下文管理器
            weights = attempt_download(weights)  # download if not found locally
        # ============加载模型以及参数================= #
        ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
        """
        两种加载模型的方式: opt.cfg / ckpt['model'].yaml
        这两种方式的区别：区别在于是否是使用resume
        如果使用resume-断点训练: 
        将opt.cfg设为空，选择ckpt['model']yaml创建模型, 且不加载anchor。
        这也影响了下面是否除去anchor的key(也就是不加载anchor), 如果resume则不加载anchor
        原因：
        使用断点训练时,保存的模型会保存anchor,所以不需要加载，
        主要是预训练权重里面保存了默认coco数据集对应的anchor，
        如果用户自定义了anchor，再加载预训练权重进行训练，会覆盖掉用户自定义的anchor。
        """
        # 加载模型
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # ***以下三行是获得anchor*** #
        # 若cfg 或 hyp.get('anchors')不为空且不使用中断训练 exclude=['anchor'] 否则 exclude=[]
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        # 将预训练模型中的所有参数保存下来，赋值给csd
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        # 判断预训练参数和新创建的模型参数有多少是相同的
        # 筛选字典中的键值对，把exclude删除
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # ***模型创建*** #
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果pretrained为ture 则会少加载两个键对（anchors, anchor_grid）
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        # 直接加载模型，ch为输入图片通道
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    amp = check_amp(model)  # check AMP

    # Freeze 冻结训练的网络层
    #     """
    #     冻结模型层,设置冻结层名字即可
    #     作用：冰冻一些层，就使得这些层在反向传播的时候不再更新权重,需要冻结的层,可以写在freeze列表中
    #     freeze为命令行参数，默认为0，表示不冻结
    #     """
    freeze = [f'model.{x}.' for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # 遍历所有层
    for k, v in model.named_parameters():
        # 为所有层的参数设置梯度
        v.requires_grad = True  # train all layers
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        # 判断是否需要冻结
        if any(x in k for x in freeze):
            LOGGER.info(f'freezing {k}')
            v.requires_grad = False

    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size  设置一次训练所选取的样本数
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({'batch_size': batch_size})

    # Optimizer
    nbs = 64  # nominal batch size 名义batch_size
    """
    nbs = 64
    batchsize = 16
    accumulate = 64 / 16 = 4
    模型梯度累计accumulate次之后就更新一次模型 相当于使用更大batch_size
    """
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减参数，防止过拟合
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf'] # 余弦退火学习率
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear 线性学习率
    # 可视化scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA # 加载ema模型和updates参数,保持ema的平滑性,现在yolov5是ema和model都保存了
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    """
    如果新设置epochs小于加载的epoch，
    则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
    """
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd  # 将预训练的相关参数从内存中删除

    # DP mode  使用单机多卡模式训练，目前一般不使用
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n'
                       'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm  多卡归一化
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    ''' =====================3.加载训练数据集==========================  '''
    # Trainloader
    train_loader, dataset = create_dataloader(train_path,
                                              imgsz,
                                              batch_size // WORLD_SIZE,
                                              gs,
                                              single_cls,
                                              hyp=hyp,
                                              augment=True,
                                              cache=None if opt.cache == 'val' else opt.cache,
                                              rect=opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=workers,
                                              image_weights=opt.image_weights,
                                              quad=opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              seed=opt.seed)
    '''
      返回一个训练数据加载器，一个数据集对象:
      训练数据加载器是一个可迭代的对象，可以通过for循环加载1个batch_size的数据
      数据集对象包括数据集的一些参数，包括所有标签值、所有的训练数据路径、每张图片的尺寸等等
    '''
    labels = np.concatenate(dataset.labels, 0)
    # 标签编号最大值
    mlc = int(labels[:, 0].max())  # max label class
    # 如果小于类别数则表示有问题
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0  验证集数据集加载
    if RANK in {-1, 0}:  # 加载验证集数据加载器
        val_loader = create_dataloader(val_path,
                                       imgsz,
                                       batch_size // WORLD_SIZE * 2,
                                       gs,
                                       single_cls,
                                       hyp=hyp,
                                       cache=None if noval else opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:  # 没有使用resume

            "计算anchor"
            # Anchors 计算默认锚框anchor与数据集标签框的高宽比
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
                '''
                参数dataset代表的是训练集，hyp['anchor_t']是从配置文件hpy.scratch.yaml读取的超参数，anchor_t:4.0
                当配置文件中的anchor计算bpr（best possible recall）小于0.98时才会重新计算anchor。
                best possible recall最大值1，如果bpr小于0.98，程序会根据数据集的label自动学习anchor的尺寸
                '''
            # 半精度
            model.half().float()  # pre-reduce anchor precision
        # 在每个训练前例行程序结束时触发所有已注册的回调
        callbacks.run('on_pretrain_routine_end', labels, names)

    # DDP mode   如果rank不等于-1,则使用DistributedDataParallel模式
    if cuda and RANK != -1:
        # local_rank为gpu编号,rank为进程,例如rank=3，local_rank=0 表示第 3 个进程内的第 1 块 GPU
        model = smart_DDP(model)

    ''' =====================4.训练==========================  '''
    '''
    4.1 初始化训练需要的模型参数
    '''
    # Model attributes  根据自己数据集的类别数和网络FPN层数设置各个损失的系数
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps)
    hyp['box'] *= 3 / nl  # scale to layers  box为预测框的损失
    hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers  cls为分类的损失
    hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers  obj为置信度损失
    hyp['label_smoothing'] = opt.label_smoothing  # 标签平滑
    model.nc = nc  # attach number of classes to model 设置模型的类别，然后将检测的类别个数保存到模型
    model.hyp = hyp  # attach hyperparameters to model  设置模型的超参数，然后将超参数保存到模型
    # 从训练的样本标签得到类别权重，然后将类别权重保存至模型
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    # 获取类别的名字，然后将分类标签保存至模型
    model.names = names

    '''
    4.2 训练热身部分
    '''
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    # 获取热身训练的迭代次数
    nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    # 初始化 map和result
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    # 设置学习率衰减所进行到的轮次，即使打断训练，使用resume接着训练也能正常衔接之前的训练进行学习率衰减
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 设置amp混合精度训练    GradScaler + autocast
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # 早停止，不更新结束训练
    stopper, stop = EarlyStopping(patience=opt.patience), False
    # 初始化损失函数
    compute_loss = ComputeLoss(model)  # init loss class
    callbacks.run('on_train_start')
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'  # 打印训练和测试输入图片分辨率
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'  # 加载图片时调用的cpu进程数
                f"Logging results to {colorstr('bold', save_dir)}\n"  # 日志目录
                f'Starting training for {epochs} epochs...')  # 从哪个epoch开始训练

    '''
    4.3 开始训练
    '''
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        '''
        告诉模型现在是训练阶段 因为BN层、DropOut层、两阶段目标检测模型等
        训练阶段阶段和预测阶段进行的运算是不同的，所以要将二者分开
        model.eval()指的是预测推断阶段
        '''
        callbacks.run('on_train_epoch_start')
        model.train()

        # Update image weights (optional, single-GPU only)  更新图片的权重
        if opt.image_weights:  # 获取图片采样的权重
            # 经过一轮训练，若哪一类的不精确度高，那么这个类就会被分配一个较高的权重，来增加它被采样的概率
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
            # 将计算出的权重换算到图片的维度，将类别的权重换算为图片的权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
            # 通过random.choices生成图片索引indices从而进行采样，这时图像会包含一些难识别的样本
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        # 初始化训练时打印的平均损失信息
        mloss = torch.zeros(3, device=device)  # mean losses
        # 分布式训练的设置
        # DDP模式打乱数据，并且dpp.sampler的随机采样数据是基于epoch+seed作为随机种子，每次epoch不同，随机种子不同
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        # 将训练数据迭代器做枚举，可以遍历出索引值
        pbar = enumerate(train_loader)
        # 训练参数的表头
        LOGGER.info(('\n' + '%11s' * 7) % ('Epoch', 'GPU_mem', 'box_loss', 'obj_loss', 'cls_loss', 'Instances', 'Size'))
        if RANK in {-1, 0}:
            # 通过tqdm创建进度条，方便训练信息的展示
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar
        # 将优化器中的所有参数梯度设为0
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run('on_train_batch_start')
            # ni: 计算当前迭代次数 iteration
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 将图片加载至设备 并做归一化
            imgs = imgs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup 热身训练
            '''
            热身训练(前nw次迭代),热身训练迭代的次数iteration范围[1:nw] 
            在前nw次迭代中, 根据以下方式选取accumulate和学习率
            '''
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                # 遍历优化器中的所有参数组
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    """
                        bias的学习率从0.1下降到基准学习率lr*lf(epoch)，
                        其他的参数学习率从0增加到lr*lf(epoch).
                        lf为上面设置的余弦退火的衰减函数
                    """
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale 设置多尺度训练，从imgsz * 0.5, imgsz * 1.5 + gs随机选取尺寸
            # imgsz: 默认训练尺寸   gs: 模型最大stride=32   [32 16 8]
            if opt.multi_scale:  # 随机改变图片的尺寸
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 下采样
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward 前向传播
            with torch.cuda.amp.autocast(amp):
                # 将图片送入网络得到一个预测结果
                pred = model(imgs)  # forward
                # 计算损失，包括分类损失，objectness损失，框的回归损失
                # loss为总损失值，loss_items为一个元组，包含分类损失，objectness损失，框的回归损失和总损失
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    # 采用DDP训练,平均不同gpu之间的梯度
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    # 如果采用collate_fn4取出mosaic4数据loss也要翻4倍
                    loss *= 4.

            # Backward  反向传播 scale为使用自动混合精度运算
            torch.use_deterministic_algorithms(False)  # cbam注意力机制
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            # 模型会对多批数据进行累积，只有达到累计次数的时候才会更新参数，再还没有达到累积次数时,loss会不断的叠加,不会被新的反传替代
            if ni - last_opt_step >= accumulate:
                '''
                 scaler.step()首先把梯度的值unscale回来，
                 如果梯度的值不是 infs 或者 NaNs, 那么调用optimizer.step()来更新权重,
                 否则，忽略step调用，从而保证权重不更新（不被破坏）
                '''

                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)  # optimizer.step 参数更新
                scaler.update()
                # 完成一次累积后,再将梯度清零,方便下一次清零
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log 打印Print一些信息 包括当前epoch、显存、损失(box、obj、cls、total)、当前batch的target的数量和图片的size等信息
            if RANK in {-1, 0}:
                # 打印显存，进行的轮次，损失，target的数量和图片的size等信息
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                # 计算显存
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                # 进度条显示以上信息
                pbar.set_description(('%11s' * 2 + '%11.4g' * 5) %
                                     (f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                # 调用Loggers中的on_train_batch_end方法，将日志记录并生成一些记录的图片
                callbacks.run('on_train_batch_end', model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler  进行学习率衰减
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        # 根据前面设置的学习率更新策略更新学习率
        scheduler.step()

        # 训练完成保存模型
        if RANK in {-1, 0}:
            # mAP
            callbacks.run('on_train_epoch_end', epoch=epoch)
            # 将model中的属性赋值给ema
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            # 判断当前epoch是否是最后一轮
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # notest: 是否只测试最后一轮  True: 只测试最后一轮   False: 每轮训练完都测试mAP
            if not noval or final_epoch:  # Calculate mAP
                """
                测试使用的是ema（指数移动平均 对模型的参数做平均）的模型
                       results: [1] Precision 所有类别的平均precision(最大f1时)
                                [1] Recall 所有类别的平均recall
                                [1] map@0.5 所有类别的平均mAP@0.5
                                [1] map@0.5:0.95 所有类别的平均mAP@0.5:0.95
                                [1] box_loss 验证集回归损失, obj_loss 验证集置信度损失, cls_loss 验证集分类损失
                       maps: [80] 所有类别的mAP@0.5:0.95
                """
                results, maps, _ = validate.run(data_dict,
                                                batch_size=batch_size // WORLD_SIZE * 2,
                                                imgsz=imgsz,
                                                half=amp,
                                                model=ema.ema,
                                                single_cls=single_cls,
                                                dataloader=val_loader,
                                                save_dir=save_dir,
                                                plots=False,
                                                callbacks=callbacks,
                                                compute_loss=compute_loss)

            # Update best mAP  更新best_fitness
            # fi: [P, R, mAP@.5, mAP@.5-.95]的一个加权值 = 0.1*mAP@.5 + 0.9*mAP@.5-.95
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            # 若当前的fitness大于最佳的fitness
            if fi > best_fitness:
                # 将最佳fitness更新为当前fitness
                best_fitness = fi
            # 保存验证结果
            log_vals = list(mloss) + list(results) + lr
            # 记录验证数据
            callbacks.run('on_fit_epoch_end', log_vals, epoch, best_fitness, fi)

            # Save model 保存模型
            #             """
            #             保存带checkpoint的模型用于inference或resuming training
            #             保存模型, 还保存了epoch, results, optimizer等信息
            #             optimizer将不会在最后一轮完成后保存
            #             model保存的是EMA的模型
            #             """
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    'epoch': epoch,
                    'best_fitness': best_fitness,
                    'model': deepcopy(de_parallel(model)).half(),
                    'ema': deepcopy(ema.ema).half(),
                    'updates': ema.updates,
                    'optimizer': optimizer.state_dict(),
                    'opt': vars(opt),
                    'git': GIT_INFO,  # {remote, branch, commit} if a git repo
                    'date': datetime.now().isoformat()}

                # Save last, best and delete 保存每轮的模型
                torch.save(ckpt, last)
                # 如果这个模型的fitness是最佳的
                if best_fitness == fi:
                    torch.save(ckpt, best)
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f'epoch{epoch}.pt')
                # 模型保存完毕 将变量从内存中删除
                del ckpt
                callbacks.run('on_model_save', last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping 早停
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    '''
    4.5 打印信息并释放显存 
    '''
    # 打印一些信息
    if RANK in {-1, 0}:
        # 训练停止 向控制台输出信息
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        # 可视化训练结果: results1.png   confusion_matrix.png 以及('F1', 'PR', 'P', 'R')曲线变化  日志信息
        for f in last, best:
            if f.exists():
                # 模型训练完后, strip_optimizer函数将optimizer从ckpt中删除
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    # 把最好的模型在验证集上跑一遍 并绘图
                    LOGGER.info(f'\nValidating {f}...')
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss)  # val best model with plots
                    if is_coco:
                        callbacks.run('on_fit_epoch_end', list(mloss) + list(results) + lr, epoch, best_fitness, fi)
        # 记录训练终止时的日志
        callbacks.run('on_train_end', last, best, epoch, results)
    # 释放显存
    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=ROOT / 'models/yolov5s_m3fd_ca_smallTarget.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/m3fd.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--project', default=ROOT / 'runs/YOLOv5(7.0)-fusion', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    '''
    2.1  检查分布式训练环境
    '''
    # Checks
    if RANK in {-1, 0}:  # 若进程编号为-1或0
        # 输出所有训练参数 / 参数以彩色的方式表现
        print_args(vars(opt))
        # 检测YOLO v5的github仓库是否更新，若已更新，给出提示
        check_git_status()
        # 检查requirements.txt所需包是否都满足
        check_requirements()

    # Resume (from specified or most recent last.pt) 判断是否断点续训
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        # opt.yaml是训练时的命令行参数文件
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        # 超参数替换，将训练时的命令行参数加载进opt参数对象中
        opt = argparse.Namespace(**d)  # replace
        # opt.cfg设置为'' 对应着train函数里面的操作(加载权重时是否加载权重里的anchor)
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        # 不使用断点续训，就从文件中读取相关参数
        # check_file （utils/general.py）的作用为查找/下载文件 并返回该文件的路径。
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        # 如果模型文件或权重文件为空，弹出警告
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            # 设置新的项目输出目录
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        # 根据opt.project生成目录，并赋值给opt.save_dir  如: runs/train/exp1
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # 判断是否分布式训练
    # DDP mode--> 支持多机多卡、分布式训练
    # 选择程序装载的位置
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 当进程内的GPU编号不为-1时，才会进入DDP
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLOv5 Multi-GPU DDP training'
        # 不能使用图片采样策略
        assert not opt.image_weights, f'--image-weights {msg}'
        # 不能使用超参数进化
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        # WORLD_SIZE表示全局的进程数
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        #  用于DDP训练的GPU数量不足
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')

    # Train  训练模式: 如果不进行超参数进化，则直接调用train()函数，开始训练
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)  遗传进化算法，边进化边训练
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'box': (1, 0.02, 0.2),  # box loss gain
            'cls': (1, 0.2, 4.0),  # cls loss gain
            'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
            'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
            'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
            'iou_t': (0, 0.1, 0.7),  # IoU training threshold
            'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
            'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
            'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        # 加载默认超参数
        with open(opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            # 如果超参数文件中没有'anchors'，则设为3
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        if opt.noautoanchor:
            del hyp['anchors'], meta['anchors']
        # 使用进化算法时，仅在最后的epoch测试和保存
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run([
                'gsutil',
                'cp',
                f'gs://{opt.bucket}/evolve.csv',
                str(evolve_csv), ])

        # 选择超参数的遗传迭代次数 默认为迭代300次
        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device, callbacks)
            callbacks = Callbacks()
            # Write mutation results
            keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss',
                    'val/obj_loss', 'val/cls_loss')
            print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {opt.evolve} generations\n'
                    f"Results saved to {colorstr('bold', save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
