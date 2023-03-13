from __future__ import division

import os
import sys
import time
import glob
import logging
from tqdm import tqdm
from random import shuffle
search = True
train = False
import torch
import torch.nn as nn
import torch.utils
from tensorboardX import SummaryWriter
import seg_metrics
import datasets
import numpy as np
import cv2
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from thop import profile
if search == True:
    import argparse
    import time
    import torch
    import numpy as np
    import torch.optim as optim
   # custom modules
    from loss import MonodepthLoss
    from utils_tools import get_model, to_device, prepare_dataloader
    # plot params
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from config import config
    from dataloader import get_train_loader

    from tools.utils.init_func import init_weight
    from tools.seg_opr.loss_opr import ProbOhemCrossEntropy2d
    from eval import SegEvaluator
    import argparse
    from supernet import Architect
    from tools.utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
    from model_search import Network_Multi_Path as Network
    from model_seg import Network_Multi_Path_Infer
    from kitti_dataset import KITTIRAWDataset
    from kitti_dataset import KITTIOdomDataset
    from torch.utils.data import DataLoader
    from dataloaders.dataloader import NewDataLoader
    from utils_new import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
        block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
    import networks
    from layers import *
if train == True:
    import os
    import sys
    import time
    import glob
    import logging
    from tqdm import tqdm

    import torch
    import torch.nn as nn
    import torch.utils
    import torch.nn.functional as F
    from tensorboardX import SummaryWriter
    import argparse
    import numpy as np
    from thop import profile

    from config_train import config

    if config.is_eval:
        config.save = 'eval-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    else:
        config.save = 'train-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    from dataloader import get_train_loader
    from datasets import Cityscapes

import argparse
from tensorboardX import SummaryWriter

from utils.init_func import init_weight
from seg_opr.loss_opr import ProbOhemCrossEntropy2d
from eval import SegEvaluator
from test import SegTester
from utils.darts_utils import create_exp_dir, save, plot_op, plot_path_width, objective_acc_lat
import seg_metrics

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
#######muilt_gpu train################################################
parser = argparse.ArgumentParser(description='cifar10 classification models')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--resume', default=None, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")

parser.add_argument('--dist-url', default='tcp://127.0.0.1:4596', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--distributed', action='store_true', help='')
######depth#############################################################


parser.add_argument("--log_dir",
                         type=str,
                         help="log directory",
                         default=os.path.join(os.path.expanduser("~"), "tmp"))

# TRAINING options

parser.add_argument("--split",
                         type=str,
                         help="which training split to use",
                         choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                         default="eigen_zhou")
parser.add_argument("--num_layers",
                         type=int,
                         help="number of resnet layers",
                         default=18,
                         choices=[18, 34, 50, 101, 152])

parser.add_argument("--png",
                         help="if set, trains from raw KITTI png files (instead of jpgs)",
                         action="store_true")

parser.add_argument("--disparity_smoothness",
                         type=float,
                         help="disparity smoothness weight",
                         default=1e-3)
parser.add_argument("--scales",
                         nargs="+",
                         type=int,
                         help="scales used in the loss",
                         default=[0, 1, 2])
parser.add_argument("--nums_scales",
                         nargs="+",
                         type=int,
                         help="scales used in the loss",
                         default=3)
parser.add_argument("--min_depth",
                         type=float,
                         help="minimum depth",
                         default=0.1)

parser.add_argument("--use_stereo",
                         help="if set, uses stereo pair for training",
                         action="store_true")
parser.add_argument("--frame_ids",
                         nargs="+",
                         type=int,
                         help="frames to load",
                         default=[0, -1, 1])

# OPTIMIZATION options


parser.add_argument("--scheduler_step_size",
                         type=int,
                         help="step size of the scheduler",
                         default=15)

# ABLATION options
parser.add_argument("--v1_multiscale",
                         help="if set, uses monodepth v1 multiscale",
                         action="store_true")
parser.add_argument("--avg_reprojection",
                         help="if set, uses average reprojection loss",
                         action="store_true")
parser.add_argument("--disable_automasking",
                         help="if set, doesn't do auto-masking",
                         action="store_true")
parser.add_argument("--predictive_mask",
                         help="if set, uses a predictive masking scheme as in Zhou et al",
                         action="store_true")
parser.add_argument("--no_ssim",
                         help="if set, disables ssim in the loss",
                         action="store_true")
parser.add_argument("--weights_init",
                         type=str,
                         help="pretrained or scratch",
                         default="pretrained",
                         choices=["pretrained", "scratch"])
parser.add_argument("--pose_model_input",
                         type=str,
                         help="how many images the pose network gets",
                         default="pairs",
                         choices=["pairs", "all"])
parser.add_argument("--pose_model_type",
                         type=str,
                         help="normal or shared",
                         default="separate_resnet",
                         choices=["posecnn", "separate_resnet", "shared"])

# SYSTEM options
parser.add_argument("--no_cuda",
                         help="if set disables CUDA",
                         action="store_true")
parser.add_argument("--num_workers",
                         type=int,
                         help="number of dataloader workers",
                         default=1)

# LOADING options
parser.add_argument("--load_weights_folder",
                         type=str,
                         help="name of model to load")
parser.add_argument("--models_to_load",
                         nargs="+",
                         type=str,
                         help="models to load",
                         default=["encoder", "depth", "pose_encoder", "pose"])

# LOGGING options
parser.add_argument("--log_frequency",
                         type=int,
                         help="number of batches between each tensorboard log",
                         default=250)
parser.add_argument("--save_frequency",
                         type=int,
                         help="number of epochs between each save",
                         default=1)

# EVALUATION options
parser.add_argument("--eval_stereo",
                         help="if set evaluates in stereo mode",
                         action="store_true")
parser.add_argument("--eval_mono",
                         help="if set evaluates in mono mode",
                         action="store_true")
parser.add_argument("--disable_median_scaling",
                         help="if set disables median scaling in evaluation",
                         action="store_true")
parser.add_argument("--pred_depth_scale_factor",
                         help="if set multiplies predictions by this number",
                         type=float,
                         default=1)
parser.add_argument("--ext_disp_to_eval",
                         type=str,
                         help="optional path to a .npy disparities file to evaluate")
parser.add_argument("--eval_split",
                         type=str,
                         default="eigen",
                         choices=[
                            "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                         help="which split to run eval on")
parser.add_argument("--save_pred_disps",
                         help="if set saves predicted disparities",
                         action="store_true")
parser.add_argument("--no_eval",
                         help="if set disables evaluation",
                         action="store_true")
parser.add_argument("--eval_eigen_to_benchmark",
                         help="if set assume we are loading eigen results from npy but "
                              "we want to evaluate using the new benchmark.",
                         action="store_true")
parser.add_argument("--eval_out_dir",
                         help="if set will output the disparities to this folder",
                         type=str)
parser.add_argument("--post_process",
                         help="if set will perform the flipping post processing "
                              "from the original monodepth paper",
                         action="store_true")
######new###########################################################
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                type=str,   help='model name', default='newcrfs_kittieigen')
parser.add_argument('--encoder',                   type=str,   help='type of encoder, base07, large07', default='large07')
parser.add_argument('--pretrain',                  type=str,   help='path of pretrained encoder', default=None)

# Dataset
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti or nyu', default='kitti')
parser.add_argument('--data_path',                 type=str,   help='path to the data', default='/home/wangshuo/Datasets/KITTI/kitti/')
parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', default='/home/wangshuo/Datasets/KITTI/data_depth_annotated/train/')
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file',default='/home/wangshuo/dou/NeWCRFs/data_splits/eigen_train_files_with_gt.txt')
parser.add_argument('--input_height',              type=int,   help='input height', default=352)
parser.add_argument('--input_width',               type=int,   help='input width',  default=1120)
parser.add_argument('--max_depth',                 type=float, help='maximum depth in estimation', default=10)

# Log and save
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='./models/')
parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                 type=int,   help='Checkpoint saving frequency in global steps', default=5000)

# Training
parser.add_argument('--weight_decay',              type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--retrain',                               help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--adam_eps',                  type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=1)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=2e-3)
parser.add_argument('--end_learning_rate',         type=float, help='end learning rate', default=-1)
parser.add_argument('--variance_focus',            type=float, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error', default=0.85)

# Preprocessing --data_path
parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)
parser.add_argument('--do_kb_crop',                            help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--use_right',                             help='if set, will randomly use right images when train on KITTI', action='store_true')

# Multi-gpu training
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                      type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                  type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',              type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                       type=int,   help='GPU id to use.', default=1)
parser.add_argument('--multiprocessing_distributed',           help='Use multi-processing distributed training to launch '
                                                                    'N processes per node, which has N GPUs. This is the '
                                                                    'fastest way to use PyTorch for either single node or '
                                                                    'multi node data parallel training', action='store_true',)
# Online eval
parser.add_argument('--do_online_eval',                        help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--data_path_eval',            type=str,   help='path to the data for online evaluation', default='/home/wangshuo/Datasets/KITTI/kitti/')
parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', default='/home/wangshuo/Datasets/KITTI/kitti/')
parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', default='/home/wangshuo/dou/NeWCRFs/data_splits/eigen_train_files_with_gt.txt')
parser.add_argument('--min_depth_eval',            type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',            type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--eigen_crop',                            help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                             help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--eval_freq',                 type=int,   help='Online evaluation frequency in global steps', default=10)
parser.add_argument('--eval_summary_directory',    type=str,   help='output directory for eval summary,'
                                                                    'if empty outputs to checkpoint folder', default='')
args = parser.parse_args()


args.gpu_devices =2,
gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
config.gpu_devices = gpu_devices

def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def adjust_learning_rate(base_lr, power, optimizer, epoch, total_epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * power

def main(search=False,train=False):
    #if search == True:
    search_train(pretrain=config.pretrain)
    #if train == True:
    #    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    #   create_exp_dir(config.save, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh'))
     #   args = parser.parse_args()
     #   ngpus_per_node = torch.cuda.device_count()
     #   args.world_size = ngpus_per_node * args.world_size
     #   mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

def search_train(pretrain=True):
    config.save = 'search-{}-{}'.format(config.save, time.strftime("%Y%m%d-%H%M%S"))
    create_exp_dir(config.save, scripts_to_save=glob.glob('*.py') + glob.glob('*.sh'))
    logger = SummaryWriter(config.save)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    assert type(pretrain) == bool or type(pretrain) == str
    update_arch = True
    if pretrain == True:
        update_arch = False
    logging.info("args = %s", str(config))
    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)

    # Model #######################################
    model = Network(config.num_classes, config.layers, ohem_criterion, Fch=config.Fch,
                    width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width)

    model = model.cuda()

    if type(pretrain) == str:
        partial = torch.load("/home/wangshuo/dou/AutoSeg_edge/search_depth/pretrain-256x512_F12.L16_batch1/weights.pt",
                             map_location='cuda:0')
        state = model.state_dict()
        pretrained_dict = {k: v for k, v in partial.items() if k in state and state[k].size() == partial[k].size()}
        state.update(pretrained_dict)
        model.load_state_dict(state)
    else:
        init_weight(model, nn.init.kaiming_normal_, nn.BatchNorm2d, config.bn_eps, config.bn_momentum, mode='fan_in',
                    nonlinearity='relu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    architect = Architect(model, config)
    # flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 2048),), verbose=False)
    # logging.info("params = %fMB, FLOPs = %fGB", params / 1e6, flops / 1e9)
    # Optimizer ###################################
    base_lr = config.lr
    parameters = []
    parameters += list(model.stem.parameters())
    parameters += list(model.cells.parameters())
    parameters += list(model.refine32.parameters())
    parameters += list(model.refine16.parameters())
    parameters += list(model.head0.parameters())
    parameters += list(model.head1.parameters())
    parameters += list(model.head2.parameters())
    parameters += list(model.head02.parameters())
    parameters += list(model.head12.parameters())
    optimizer = torch.optim.SGD(
        parameters,
        lr=base_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay)

    # lr policy ##############################
    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)

    # New data loader ###########################

    dataloader = NewDataLoader(args, 'train')
    dataloader_eval = NewDataLoader(args, 'online_eval')

    #eval######################
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    filenames = readlines(os.path.join(splits_dir, args.eval_split, "test_files.txt"))
    if update_arch:
        logger.add_scalar("arch/latency_weight", config.latency_weight[1], 0)
        logging.info("arch_latency_weight = " + str(config.latency_weight[1]))
        logger.add_scalar("arch/flops_weight", config.flops_weight[1], 0)
        logging.info("arch_flops_weight = " + str(config.flops_weight[1]))

    tbar = tqdm(range(config.nepochs), ncols=80)
    valid_mIoU_history = []
    FPSs_history = []
    FPSs = []
    latency_supernet_history = []
    latency_weight_history = []
    valid_names = ["8s", "16s", "32s", "8s_32s", "16s_32s"]
    arch_names = {0: "teacher", 1: "student"}
    for epoch in tbar:
        logging.info(pretrain)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))

        logging.info("update arch: " + str(update_arch))
        loss = MonodepthLoss(
            n=2,
            SSIM_w=0.85,
            disp_gradient_w=0.1, lr_w=1).to(device)


        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train(pretrain, dataloader, model, architect, loss, ohem_criterion, optimizer, lr_policy, logger, epoch,args, update_arch=update_arch)
        torch.cuda.empty_cache()
        lr_policy.step()
        #val###
        # log less frequently after the first 2000 steps to save time & disk space
        #####val###############

        val(args,model,dataloader_eval,args.gpu_devices,1,epoch)
        save(model, os.path.join(config.save, 'weights.pt'))
        for idx, arch_name in enumerate(model._arch_names):
            state = {}
            for name in arch_name['fais']:
                state[name] = getattr(model, name)
            for name in arch_name['mjus']:
                state[name] = getattr(model, name)
            for name in arch_name['thetas']:
                state[name] = getattr(model, name)
            fps0, fps1 = arch_logging(model, config, logger, epoch)
            FPSs.append([fps0, fps1])
            state["latency02"] = 1000. / fps0
            state["latency12"] = 1000. / fps1
            torch.save(state, os.path.join(config.save, "arch_%d.pt" % epoch))
            torch.save(state, os.path.join(config.save, "arch.pt"))
        idx = 0
        if config.latency_weight[idx] > 0:
            if (int(FPSs[idx][0] >= config.FPS_max[idx]) + int(FPSs[idx][1] >= config.FPS_max[idx])) >= 1:
                architect.latency_weight[idx] /= 2
            elif (int(FPSs[idx][0] <= config.FPS_min[idx]) + int(FPSs[idx][1] <= config.FPS_min[idx])) > 0:
                architect.latency_weight[idx] *= 2
            logger.add_scalar("arch/latency_weight_%s" % arch_names[idx], architect.latency_weight[idx], epoch + 1)
            logging.info("arch_latency_weight_%s = " % arch_names[idx] + str(architect.latency_weight[idx]))
        if config.flops_weight[idx] > 0:
            if (int(FPSs[idx][0] >= config.FPS_max[idx]) + int(FPSs[idx][1] >= config.FPS_max[idx])) >= 1:
                architect.flops_weight[idx] /= 2
            elif (int(FPSs[idx][0] <= config.FPS_min[idx]) + int(FPSs[idx][1] <= config.FPS_min[idx])) > 0:
                architect.flops_weight[idx] *= 2
            logger.add_scalar("arch/latency_weight_%s" % arch_names[idx], architect.flops_weight[idx],
                              epoch + 1)
            logging.info("arch_latency_weight_%s = " % arch_names[idx] + str(architect.flops_weight[idx]))


def main_worker(gpu, ngpus_per_node, args):
    logger = SummaryWriter(config.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(config.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info("args = %s", str(config))
    # dist###########################
    args.gpu = gpu
    ngpus_per_node = torch.cuda.device_count()
    print("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)

    # preparation ################
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # config network and criterion ################
    min_kept = int(config.batch_size * config.image_height * config.image_width // (16 * config.gt_down_sampling ** 2))
    ohem_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=min_kept, use_weight=False)
    distill_criterion = nn.KLDivLoss().cuda()

    # data loader ###########################
    if config.is_test:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_eval_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}
    else:
        data_setting = {'img_root': config.img_root_folder,
                        'gt_root': config.gt_root_folder,
                        'train_source': config.train_source,
                        'eval_source': config.eval_source,
                        'test_source': config.test_source,
                        'down_sampling': config.down_sampling}

    train_loader = get_train_loader(config, Cityscapes)

    # Model #######################################
    models = []
    evaluators = []
    testers = []
    lasts = []
    start_epoch = -1

    for idx, arch_idx in enumerate(config.arch_idx):
        if config.load_epoch == "last":
            state = torch.load(os.path.join(config.load_path, "arch_%d.pt" % arch_idx), map_location='cuda:0')
        else:
            state = torch.load(os.path.join(config.load_path, "arch_%d_%d.pt" % (arch_idx, int(config.load_epoch))))

        model = Network(
            [state["alpha_%d_0" % arch_idx].detach(), state["alpha_%d_1" % arch_idx].detach(),
             state["alpha_%d_2" % arch_idx].detach()],
            [None, state["beta_%d_1" % arch_idx].detach(), state["beta_%d_2" % arch_idx].detach()],
            [state["ratio_%d_0" % arch_idx].detach(), state["ratio_%d_1" % arch_idx].detach(),
             state["ratio_%d_2" % arch_idx].detach()],
            num_classes=config.num_classes, layers=config.layers, Fch=config.Fch,
            width_mult_list=config.width_mult_list, stem_head_width=config.stem_head_width[idx])
        torch.cuda.set_device(args.rank)
        model.cuda(args.rank)
        config.num_workers = int(config.num_workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.rank], find_unused_parameters=True)

        mIoU02 = state["mIoU02"];
        latency02 = state["latency02"];
        obj02 = objective_acc_lat(mIoU02, latency02)
        mIoU12 = state["mIoU12"];
        latency12 = state["latency12"];
        obj12 = objective_acc_lat(mIoU12, latency12)
        if obj02 > obj12:
            last = [2, 0]
        else:
            last = [2, 1]
        lasts.append(last)
        model.module.build_structure(last)
        logging.info("net: " + str(model))
        for b in last:
            if len(config.width_mult_list) > 1:
                plot_op(getattr(model.module, "ops%d" % b), getattr(model.module, "path%d" % b),
                        width=getattr(model.module, "widths%d" % b), head_width=config.stem_head_width[idx][1],
                        F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png" % (arch_idx, b)),
                                                   bbox_inches="tight")
            else:
                plot_op(getattr(model.module, "ops%d" % b), getattr(model.module, "path%d" % b),
                        F_base=config.Fch).savefig(os.path.join(config.save, "ops_%d_%d.png" % (arch_idx, b)),
                                                   bbox_inches="tight")
        plot_path_width(model.module.lasts, model.module.paths, model.module.widths).savefig(
            os.path.join(config.save, "path_width%d.png" % arch_idx))
        plot_path_width([2, 1, 0], [model.module.path2, model.module.path1, model.module.path0],
                        [model.module.widths2, model.module.widths1, model.module.widths0]).savefig(
            os.path.join(config.save, "path_width_all%d.png" % arch_idx))
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)
        # logging.info("ops:" + str(model.module.mops))
        # logging.info("path:" + str(model.module.paths))
        # logging.info("last:" + str(model.module.lasts))
        model = model.cuda(args.rank)
        init_weight(model, nn.init.kaiming_normal_, torch.nn.BatchNorm2d, config.bn_eps, config.bn_momentum,
                    mode='fan_in', nonlinearity='relu')
        if arch_idx == 0 and len(config.arch_idx) > 1:
            model = smp.Unet(encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                             encoder_weights="imagenet",
                             # use `imagenet` pre-trained weights for encoder initialization
                             in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                             classes=19,  # model output channels (number of classes in your dataset)
                             ).cuda()

        elif config.is_eval:
            partial = torch.load(os.path.join(config.eval_path, "weights%d.pt" % arch_idx), map_location='cuda:0')
            state = model.module.state_dict()
            pretrained_dict = {k: v for k, v in partial.items() if k in state}
            state.update(pretrained_dict)
            model.module.load_state_dict(state)

        evaluator = SegEvaluator(Cityscapes(data_setting, 'val', None), config.num_classes, config.image_mean,
                                 config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0,
                                 config=config,
                                 verbose=False, save_path=None, show_image=False, show_prediction=False)
        evaluators.append(evaluator)
        tester = SegTester(Cityscapes(data_setting, 'test', None), config.num_classes, config.image_mean,
                           config.image_std, model, config.eval_scale_array, config.eval_flip, 0, out_idx=0,
                           config=config,
                           verbose=False, save_path=None, show_prediction=False)
        testers.append(tester)

        # Optimizer ###################################
        base_lr = config.lr
        if arch_idx == 1 or len(config.arch_idx) == 1:
            # optimize teacher solo OR student (w. distill from teacher)
            optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        models.append(model)

        if config.RESUME:
            path_checkpoint = "./checkpoint/ckpt_best_17.pth"  # 断点路径

            checkpoint = torch.load(path_checkpoint)  # 加载断点
            start_epoch = checkpoint['epoch']  # 设置开始的epoch
            if arch_idx == 1 or len(config.arch_idx) == 1:
                optimizer.load_state_dict(checkpoint['optimizer'])
                partial = torch.load(os.path.join(
                    '/home/wangshuo/douzi/pytorch-multigpu/train_dist/train-512x1024_student_batch20-20211128-202309',
                    "weights%d.pt" % arch_idx), map_location='cuda:0')
                state = model.module.state_dict()
                pretrained_dict = {k: v for k, v in partial.items() if k in state}
                state.update(pretrained_dict)
                model.module.load_state_dict(state)

    # Cityscapes ###########################################
    if config.is_eval:
        logging.info(config.load_path)
        logging.info(config.eval_path)
        logging.info(config.save)
        with torch.no_grad():
            if config.is_test:
                # test
                print("[test...]")
                with torch.no_grad():
                    test_student(0, models, testers, logger)
            else:
                # validation
                print("[validation...]")
                valid_mIoUs = infer_student(models, evaluators, logger)
                for idx, arch_idx in enumerate(config.arch_idx):
                    if arch_idx == 1:
                        logger.add_scalar("mIoU/val_student", valid_mIoUs[idx], 0)
                        logging.info("student's valid_mIoU %.3f" % (valid_mIoUs[idx]))
        exit(0)
    vaild_student_max = 0
    tbar = tqdm(range(config.nepochs), ncols=80)
    for epoch in range(start_epoch + 1, 300):
        logging.info(config.load_path)
        logging.info(config.save)
        logging.info("lr: " + str(optimizer.param_groups[0]['lr']))
        # training
        tbar.set_description("[Epoch %d/%d][train...]" % (epoch + 1, config.nepochs))
        train_mIoUs = train_student(train_loader, models, ohem_criterion, distill_criterion, optimizer, logger, epoch)
        torch.cuda.empty_cache()
        for idx, arch_idx in enumerate(config.arch_idx):
            if arch_idx == 1:
                logger.add_scalar("mIoU/train_student", train_mIoUs[idx], epoch)
                logging.info("student's train_mIoU %.3f" % (train_mIoUs[idx]))
        adjust_learning_rate(base_lr, 0.992, optimizer, epoch + 1, config.nepochs)

        # validation
        if not config.is_test and ((epoch + 1) % 1 == 0 or epoch == 0):
            tbar.set_description("[Epoch %d/%d][validation...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                valid_mIoUs = infer_student(models, evaluators, logger)
                for idx, arch_idx in enumerate(config.arch_idx):
                    if arch_idx == 1:
                        logger.add_scalar("mIoU/val_student", valid_mIoUs[idx], epoch)
                        logging.info("student's valid_mIoU %.3f" % (valid_mIoUs[idx]))
                    save(models[1], os.path.join(config.save, "weights%d.pt" % 1))
                    # resume
                    checkpoint = {'optimizer': optimizer.state_dict(),
                                  'epoch': epoch}
                    torch.save(checkpoint, './checkpoint/ckpt_best_%d.pth' % epoch)
        # test
        if config.is_test and (epoch + 1) >= 250 and (epoch + 1) % 10 == 0:
            tbar.set_description("[Epoch %d/%d][test...]" % (epoch + 1, config.nepochs))
            with torch.no_grad():
                test_student(epoch, models, testers, logger)

        save(models[1], os.path.join(config.save, "weights%d.pt" % 1))
        for idx, arch_name in enumerate(model._arch_names):
            state = {}
            for name in arch_name['fais']:
                state[name] = getattr(model, name)
            for name in arch_name['mjus']:
                state[name] = getattr(model, name)
            for name in arch_name['thetas']:
                state[name] = getattr(model, name)
            state["mIoU02"] = valid_mIoUs[3]
            state["mIoU12"] = valid_mIoUs[4]
            state["latency02"] = 1000. / fps0
            state["latency12"] = 1000. / fps1
            torch.save(state, os.path.join(config.save, "arch_%d.pt" % epoch))
            torch.save(state, os.path.join(config.save, "arch.pt"))

        for idx in range(len(config.latency_weight)):
            if config.latency_weight[idx] > 0:
                if (int(FPSs[idx][0] >= config.FPS_max[idx]) + int(FPSs[idx][1] >= config.FPS_max[idx])) >= 1:
                    architect.latency_weight[idx] /= 2
                elif (int(FPSs[idx][0] <= config.FPS_min[idx]) + int(FPSs[idx][1] <= config.FPS_min[idx])) > 0:
                    architect.latency_weight[idx] *= 2
                logger.add_scalar("arch/latency_weight_%s" % arch_names[idx], architect.latency_weight[idx],
                                  epoch + 1)
                logging.info("arch_latency_weight_%s = " % arch_names[idx] + str(architect.latency_weight[idx]))
        for idx in range(len(config.flops_weight)):
            if config.flops_weight[idx] > 0:
                if (int(FPSs[idx][0] >= config.FPS_max[idx]) + int(FPSs[idx][1] >= config.FPS_max[idx])) >= 1:
                    architect.flops_weight[idx] /= 2
                elif (int(FPSs[idx][0] <= config.FPS_min[idx]) + int(FPSs[idx][1] <= config.FPS_min[idx])) > 0:
                    architect.flops_weight[idx] *= 2
                logger.add_scalar("arch/latency_weight_%s" % arch_names[idx], architect.flops_weight[idx],
                                  epoch + 1)
                logging.info("arch_latency_weight_%s = " % arch_names[idx] + str(architect.flops_weight[idx]))
def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M

def generate_images_pred(self, inputs, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    for scale in self.opt.scales:
        disp = outputs[("disp", scale)]
        if self.opt.v1_multiscale:
            source_scale = scale
        else:
            disp = F.interpolate(
                disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
            source_scale = 0

        _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

        outputs[("depth", 0, scale)] = depth

        for i, frame_id in enumerate(self.opt.frame_ids[1:]):

            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            # from the authors of https://arxiv.org/abs/1712.00175
            if self.opt.pose_model_type == "posecnn":

                axisangle = outputs[("axisangle", 0, frame_id)]
                translation = outputs[("translation", 0, frame_id)]

                inv_depth = 1 / depth
                mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                T = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

            cam_points = self.backproject_depth[source_scale](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[source_scale](
                cam_points, inputs[("K", source_scale)], T)

            outputs[("sample", frame_id, scale)] = pix_coords

            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")

            if not self.opt.disable_automasking:
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]


##########dips###########################

def compute_reprojection_loss(self, pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    self.ssim = SSIM()
    if(target.dim()==3):
       target = torch.randn(self.batch_size,target.size(0), target.size(1), target.size(2))
    if (pred.dim() == 3):
       pred = torch.randn(self.batch_size, pred.size(0), pred.size(1), pred.size(2))
    target = target.cuda()
    pred = pred.cuda()
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)
    ssim_loss = self.ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss

def get_smooth_loss(self,disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    if(img.dim()==3):
      img = torch.randn(self.batch_size,img.size(0),img.size(1),img.size(2))
    img = img.cuda()
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


def compute_losses(self, inputs, outputs):
    """Compute the reprojection and smoothness losses for a minibatch
    """
    losses = {}
    total_loss = 0

    for scale in self.scales:
        loss = 0
        reprojection_losses = []

        if self.v1_multiscale:
            source_scale = scale
        else:
            source_scale = 0

        disp = outputs[("disp", scale)]
        color = inputs[("color", 0, scale)]
        target = inputs[("color", 0, source_scale)]

        for frame_id in self.frame_ids[1:]:
            pred = outputs[("color", frame_id, scale)]
            reprojection_losses.append(compute_reprojection_loss(self,pred, target))

        reprojection_losses = torch.cat(reprojection_losses, 1)

        if not self.disable_automasking:
            identity_reprojection_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = inputs[("color", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    compute_reprojection_loss(self,pred, target))

            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

            if self.avg_reprojection:
                identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
            else:
                # save both images, and do min all at once below
                identity_reprojection_loss = identity_reprojection_losses

        elif self.predictive_mask:
            # use the predicted mask
            mask = outputs["predictive_mask"]["disp", scale]
            if not self.v1_multiscale:
                mask = F.interpolate(
                    mask, [self.height, self.width],
                    mode="bilinear", align_corners=False)

            reprojection_losses *= mask

            # add a loss pushing mask to 1 (using nn.BCELoss for stability)
            weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda())
            loss += weighting_loss.mean()

        if self.avg_reprojection:
            reprojection_loss = reprojection_losses.mean(1, keepdim=True)
        else:
            reprojection_loss = reprojection_losses

        if not self.disable_automasking:
            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape, device=0) * 0.00001

            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
        else:
            combined = reprojection_loss

        if combined.shape[1] == 1:
            to_optimise = combined
        else:
            to_optimise, idxs = torch.min(combined, dim=1)

        if not self.disable_automasking:
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

        loss += to_optimise.mean()

        mean_disp = disp.mean(2, True).mean(3, True)
        norm_disp = disp / (mean_disp + 1e-7)
        smooth_loss = get_smooth_loss(args,norm_disp, color)

        loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= self.nums_scales
    losses["loss"] = total_loss
    return losses

def online_eval(args,model, dataloader_eval, gpu, ngpus, post_process=False):
    eval_measures = torch.zeros(10).cuda()
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_sample_batched['image'].cuda())
            gt_depth = eval_sample_batched['depth']
            #sprint(gt_depth)
            has_valid_depth = eval_sample_batched['has_valid_depth']
            if not has_valid_depth:
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(args, image, False, True,False)
            if post_process:
                image_flipped = flip_lr(image)
                pred_depth_flipped = model(args, image_flipped, False, True,False)
                pred_depth = post_process_depth(pred_depth, pred_depth_flipped)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        if eval_sample_batched['has_valid_depth']:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height), int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height), int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                elif args.dataset == 'nyu':
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda()
        eval_measures[9] += 1

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)), ', post_process: ', post_process)
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.4f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.4f}'.format(eval_measures_cpu[8]))
        return eval_measures_cpu

    return None


def val(self,model, dataloader_eval, gpu, ngpus_per_node,epoch):
    """Validate the model on a single minibatch
    """
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)
    eval_summary_path = os.path.join(args.log_directory, args.model_name, 'eval')
    eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)
    with torch.no_grad():
        eval_measures = online_eval(args, model, dataloader_eval, gpu, ngpus_per_node, post_process=True)
    if eval_measures is not None:
        for i in range(9):
            eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(epoch))
            measure = eval_measures[i]
            is_best = False
            if i < 6 and measure < best_eval_measures_lower_better[i]:
                old_best = best_eval_measures_lower_better[i].item()
                best_eval_measures_lower_better[i] = measure.item()
                is_best = True
            elif i >= 6 and measure > best_eval_measures_higher_better[i - 6]:
                old_best = best_eval_measures_higher_better[i - 6].item()
                best_eval_measures_higher_better[i - 6] = measure.item()
                is_best = True
            if is_best:
                old_best_step = best_eval_steps[i]
                old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                model_path = args.log_directory + '/' + args.model_name + old_best_name
                if os.path.exists(model_path):
                    command = 'rm {}'.format(model_path)
                    os.system(command)
                best_eval_steps[i] = epoch
                model_save_name = '/model-{}-best_{}_{:.5f}'.format(epoch, eval_metrics[i], measure)
                print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                checkpoint = {'global_step': epoch,
                              'model': model.state_dict(),
                              'best_eval_measures_higher_better': best_eval_measures_higher_better,
                              'best_eval_measures_lower_better': best_eval_measures_lower_better,
                              'best_eval_steps': best_eval_steps
                              }
                torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
        eval_summary_writer.flush()

def train(pretrain,loader, model, architect, loss, criterion, optimizer, lr_policy, logger, epoch,arg, update_arch=True):
    model.train()
    losses = []
    train_losses = []
    best_loss = float('Inf')
    best_val_loss = float('Inf')
    running_val_loss = 0.0
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    running_loss = 0.0
    for batch_idx, inputs in enumerate(loader.data):
        image = torch.autograd.Variable(inputs['image'].cuda())
        depth_gt = torch.autograd.Variable(inputs['depth'].cuda())
        c_time = time.time()
        optimizer.zero_grad()
        for key, ipt in inputs.items():
            inputs[key] = ipt.cuda()
        if update_arch:
            # get a random minibatch from the search queue with replacement
            pbar.set_description("[Arch Step %d/%d]" % (batch_idx + 1, len(loader.data)))
            loss_arch = architect.step(arg,image, inputs, depth_gt, inputs)
            print('loss_arch/train',loss_arch)
            if (batch_idx+1) % 10 == 0:
                logger.add_scalar('loss_arch/train', loss_arch, epoch*len(pbar)+batch_idx)
                logger.add_scalar('arch/latency_supernet', architect.latency_supernet, epoch*len(pbar)+batch_idx)
                logger.add_scalar('arch/flops_supernet', architect.flops_supernet, epoch * len(pbar) + batch_idx)
        optimizer.zero_grad()
        
        disps = model(arg,image, inputs,False,False)
        mask = depth_gt > 1.0
        silog_criterion = silog_loss(variance_focus=arg.variance_focus).cuda()
        loss = silog_criterion.forward(arg, disps, depth_gt, mask.to(torch.bool))
        print('loss/train', loss)
        loss.backward()
        optimizer.step()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description("[Step %d/%d]" % (batch_idx + 1, len(loader.data)))
    # Estimate loss per image
    print(
        'Epoch:',
        epoch + 1,
        "train_arch_loss:",
        loss_arch,
        'train_model_loss:',
        running_loss,
        'time:',
        round(time.time() - c_time, 3),
        's',
    )
    torch.cuda.empty_cache()
    # del loss
    # if update_arch: del loss_arch

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def evaluate(opt,model):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    STEREO_SCALE_FACTOR = 5.4
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")

    if opt.ext_disp_to_eval is None:

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        dataset = datasets.KITTIRAWDataset("/home/wangshuo/dou/MonoDepth-PyTorch/data/kitti/", filenames,
                                           192, 640,
                                           [0], 4, is_train=False)
        dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                                pin_memory=True, drop_last=False)
        # data loader ###########################

        pred_disps = []
        print("-> Computing predictions with size {}x{}")
        with torch.no_grad():
            for data in dataloader:
                input_color = data[("color", 0, 0)].cuda()
                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                output = model(opt,input_color,data,True,True)

                pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)


        pred_disps = np.concatenate(pred_disps)
        print(pred_disps.shape[0])

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    return mean_errors



def infer(pretrain,val_loader, model, architect, loss, criterion, optimizer, lr_policy, logger, epoch,arg, update_arch=True):
    losses = []
    val_losses = []
    mIoUs = []
    best_loss = float('Inf')
    best_val_loss = float('Inf')
    dataloader_model = iter(val_loader)
    minibatch = dataloader_model.next()
    running_val_loss = 0.0
    for data in val_loader:
        left = data['left_image']
        right = data['right_image']
        imgs = minibatch['left_image']
        target = minibatch['right_image']
        imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        disps = model(arg, inputs["color_aug", 0, 0], inputs)
        losses = compute_losses(arg, inputs, disps)
        val_losses.append(lossdepth.item())
        running_val_loss += lossdepth.item()
    running_val_loss /= val_n_img / 1
    print('Val_loss:', running_val_loss)
    mIoUs = [running_val_loss,0,0,0,0]
    if FPS:
        fps0, fps1 = arch_logging(model, config, logger, epoch)
        return mIoUs, fps0, fps1
    else:
        return mIoUs



def train_student(train_loader, models, criterion, distill_criterion, optimizer, logger, epoch):
    if len(models) == 1:
        # train teacher solo
        models[0].train()
        models[0].cuda()
    else:
        # train student (w. distill from teacher)
        models[0].eval()
        models[1].train()

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format, ncols=80)
    dataloader = iter(train_loader)

    metrics = [ seg_metrics.Seg_Metrics(n_classes=config.num_classes) for _ in range(len(models)) ]
    lamb = 0.2
    for step in pbar:
        optimizer.zero_grad()

        minibatch = dataloader.next()
        imgs = minibatch['data']
        imgs = imgs.type(torch.FloatTensor)
        target = minibatch['label']
        #imgs = imgs.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        logits_list = []
        #hardTarget
        loss = 0
        #softTarget
        loss_kl = 0
        description = ""
        for idx, arch_idx in enumerate(config.arch_idx):
            model = models[idx]
            if arch_idx == 0 and len(models) > 1:
                with torch.no_grad():
                    imgs = imgs.cuda(non_blocking=True)
                    logits8 = model(imgs)
                    logits_list.append(logits8)
            else:
                imgs = imgs.cuda(non_blocking=True)
                logits8, logits16, logits32 = model(imgs)
                logits_list.append(logits8)
                logits8.cuda()
                logits16.cuda()
                logits32.cuda()
                loss = loss + criterion(logits8, target)
                loss = loss + lamb * criterion(logits16, target)
                loss = loss + lamb * criterion(logits32, target)
                loss.cuda()
                if len(logits_list) > 1:
                    distill_criterion = distill_criterion.cuda()
                    loss = loss + distill_criterion(F.softmax(logits_list[1], dim=1).log().cuda(), F.softmax(logits_list[0], dim=1).cuda())

            metrics[idx].update(logits8.data, target)
        description += "[mIoU%d: %.3f]"%(1, metrics[1].get_scores())

        pbar.set_description("[Step %d/%d]"%(step + 1, len(train_loader)) + description)
        logger.add_scalar('loss/train', loss+loss_kl, epoch*len(pbar)+step)

        loss.backward()
        optimizer.step()

    return [ metric.get_scores() for metric in metrics ]

def infer_student(models, evaluators, logger):
    mIoUs = []
    for model, evaluator in zip(models, evaluators):
        model.eval()
        _, mIoU = evaluator.run_online()
        #_, mIoU = evaluator.run_online_multiprocess()
        mIoUs.append(mIoU)
    return mIoUs

def test_student(epoch, models, testers, logger):
    for idx, arch_idx in enumerate(config.arch_idx):
        if arch_idx == 0: continue
        model = models[idx]
        tester = testers[idx]
        os.system("mkdir %s"%os.path.join(os.path.join(os.path.realpath('.'), config.save, "test")))
        model.eval()
        tester.run_online()
        os.system("mv %s %s"%(os.path.join(os.path.realpath('.'), config.save, "test"), os.path.join(os.path.realpath('.'), config.save, "test_%d_%d"%(arch_idx, epoch))))

def arch_logging(model, args, logger, epoch):
    input_size = (1, 3, 256, 512)
    net = Network_Multi_Path_Infer(
        [getattr(model, model._arch_names[model.arch_idx]["fais"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["fais"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["fais"][2]).clone().detach()],
        [None, getattr(model, model._arch_names[model.arch_idx]["mjus"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["mjus"][1]).clone().detach()],
        [getattr(model, model._arch_names[model.arch_idx]["thetas"][0]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["thetas"][1]).clone().detach(), getattr(model, model._arch_names[model.arch_idx]["thetas"][2]).clone().detach()],
        num_classes=model._num_classes, layers=model._layers, Fch=model._Fch, width_mult_list=model._width_mult_list, stem_head_width=model._stem_head_width[0])

    plot_op(net.ops0, net.path0, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops0_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops1, net.path1, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops1_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)
    plot_op(net.ops2, net.path2, F_base=args.Fch).savefig("table.png", bbox_inches="tight")
    logger.add_image("arch/ops2_arch%d"%model.arch_idx, np.swapaxes(np.swapaxes(plt.imread("table.png"), 0, 2), 1, 2), epoch)

    net.build_structure([2, 0])
    net = net.cuda()
    net.eval()
    latency0, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps0_arch%d"%model.arch_idx, 1000./latency0, epoch)
    logger.add_figure("arch/path_width_arch%d_02"%model.arch_idx, plot_path_width([2, 0], [net.path2, net.path0], [net.widths2, net.widths0]), epoch)

    net.build_structure([2, 1])
    net = net.cuda()
    net.eval()
    latency1, _ = net.forward_latency(input_size[1:])
    logger.add_scalar("arch/fps1_arch%d"%model.arch_idx, 1000./latency1, epoch)
    logger.add_figure("arch/path_width_arch%d_12"%model.arch_idx, plot_path_width([2, 1], [net.path2, net.path1], [net.widths2, net.widths1]), epoch)

    return 1000./latency0, 1000./latency1


if __name__ == '__main__':
    main(search=False,train=True)
