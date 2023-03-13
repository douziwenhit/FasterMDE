import torch
import torch.nn as nn
import torch.nn.functional as F
from operator_pool import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
# from utils.darts_utils import drop_path, compute_speed, compute_speed_tensorrt
from pdb import set_trace as bp
from seg_oprs import Head
import numpy as np
import networks_mono2
from layers import *
from collections import OrderedDict
from loss import MonodepthLoss
from mmcv.cnn import ConvModule
import torch.utils.checkpoint as checkpoint
import collections.abc
from itertools import repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from utils_new import post_process_depth, flip_lr, silog_loss, compute_errors, eval_metrics, \
                       block_print, enable_print, normalize_result, inv_normalize, convert_arg_line_to_args
###############depth#################################
def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)

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



def predict_poses(self, inputs, features):
    """Predict poses between input frames for monocular sequences.
    """
    a = []
    outputs = {}
    model_pose_encoder = networks_mono2.ResnetEncoder(self.num_layers, self.weights_init == "pretrained",num_input_images=2)

    model_pose = networks_mono2.PoseDecoder(model_pose_encoder.num_ch_enc,num_input_features=1,num_frames_to_predict_for=2)


    # In this setting, we compute the pose to each source frame via a
    # separate forward pass through the pose network.

    # select what features the pose network takes as input

    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.frame_ids}

    for f_i in self.frame_ids[1:]:
        if f_i != "s":
            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs[0] = torch.randn((self.batch_size,3,192,640))
            pose_inputs[1] = torch.randn((self.batch_size,3,192,640))
            pose_inputs = model_pose_encoder(torch.cat(pose_inputs, 1))
            a.append(pose_inputs)
            axisangle, translation = model_pose(a[f_i])
            outputs[("axisangle", 0, f_i)] = axisangle
            outputs[("translation", 0, f_i)] = translation

            # Invert the matrix if the frame id is negative
            outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))
    return outputs

class conv(nn.Module):
    def __init__(self, num_in_layers=128, num_out_layers=256, kernel_size=3, stride=1):
        super(conv, self).__init__()
        self.kernel_size = kernel_size
        self.conv_base = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=kernel_size, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1)/2))
        p2d = (p, p, p, p)
        x = self.conv_base(F.pad(x, p2d))
        x = self.normalize(x)
        return F.elu(x, inplace=True)


class convblock(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size):
        super(convblock, self).__init__()
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, kernel_size, 2)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)


class maxpool(nn.Module):
    def __init__(self, kernel_size):
        super(maxpool, self).__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        p = int(np.floor((self.kernel_size-1) / 2))
        p2d = (p, p, p, p)
        return F.max_pool2d(F.pad(x, p2d), self.kernel_size, stride=2)


class resconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 1, 1)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, stride)
        self.conv3 = nn.Conv2d(num_out_layers, 4*num_out_layers, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(num_in_layers, 4*num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(4*num_out_layers)

    def forward(self, x):
        # do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)
        if do_proj:
            shortcut = self.conv4(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


class resconv_basic(nn.Module):
    # for resnet18
    def __init__(self, num_in_layers, num_out_layers, stride):
        super(resconv_basic, self).__init__()
        self.num_out_layers = num_out_layers
        self.stride = stride
        self.conv1 = conv(num_in_layers, num_out_layers, 3, stride)
        self.conv2 = conv(num_out_layers, num_out_layers, 3, 1)
        self.conv3 = nn.Conv2d(num_in_layers, num_out_layers, kernel_size=1, stride=stride)
        self.normalize = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        #         do_proj = x.size()[1] != self.num_out_layers or self.stride == 2
        do_proj = True
        shortcut = []
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        if do_proj:
            shortcut = self.conv3(x)
        else:
            shortcut = x
        return F.elu(self.normalize(x_out + shortcut), inplace=True)


def resblock(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks - 1):
        layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    layers.append(resconv(4 * num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


def resblock_basic(num_in_layers, num_out_layers, num_blocks, stride):
    layers = []
    layers.append(resconv_basic(num_in_layers, num_out_layers, stride))
    for i in range(1, num_blocks):
        layers.append(resconv_basic(num_out_layers, num_out_layers, 1))
    return nn.Sequential(*layers)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super(upconv, self).__init__()
        self.scale = scale
        self.conv1 = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=True)
        return self.conv1(x)


class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)




# https://github.com/YongfeiYan/Gumbel_Softmax_VAE/blob/master/gumbel_softmax_vae.py
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class MixedOp(nn.Module):

    def __init__(self, C_in, C_out, stride=1, width_mult_list=[1.]):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._width_mult_list = width_mult_list
        for primitive in PRIMITIVES:
            op = OPS[primitive](C_in, C_out, stride, True, width_mult_list=width_mult_list)
            self._ops.append(op)

    def set_prun_ratio(self, ratio):
        for op in self._ops:
            op.set_ratio(ratio)

    def forward(self, x, weights, thetas):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(thetas[0], torch.Tensor):
            ratio0 = self._width_mult_list[thetas[0].argmax()]
            r_score0 = thetas[0][thetas[0].argmax()]
        else:
            ratio0 = thetas[0]
            r_score0 = 1.
        if isinstance(thetas[1], torch.Tensor):
            ratio1 = self._width_mult_list[thetas[1].argmax()]
            r_score1 = thetas[1][thetas[1].argmax()]
        else:
            ratio1 = thetas[1]
            r_score1 = 1.
        self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(weights, self._ops):
            op(x).cuda()
            result = result + op(x) * w * r_score0 * r_score1 #  每一次的结果相加
        return result

    def forward_latency(self, size, weights, thetas):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(thetas[0], torch.Tensor):
            ratio0 = self._width_mult_list[thetas[0].argmax()]
            r_score0 = thetas[0][thetas[0].argmax()]
        else:
            ratio0 = thetas[0]
            r_score0 = 1.
        if isinstance(thetas[1], torch.Tensor):
            ratio1 = self._width_mult_list[thetas[1].argmax()]
            r_score1 = thetas[1][thetas[1].argmax()]
        else:
            ratio1 = thetas[1]
            r_score1 = 1.
        self.set_prun_ratio((ratio0, ratio1))
        for w, op in zip(weights, self._ops):
            latency, size_out = op.forward_latency(size)
            result = result + latency * w * r_score0 * r_score1
        return result, size_out

    def forward_flops(self, size, fai, ratio):
        # int: force #channel; tensor: arch_ratio; float(<=1): force width
        result = 0
        if isinstance(ratio[0], torch.Tensor):
            ratio0 = self._width_mult_list[ratio[0].argmax()]
            r_score0 = ratio[0][ratio[0].argmax()]
        else:
            ratio0 = ratio[0]
            r_score0 = 1.
        if isinstance(ratio[1], torch.Tensor):
            ratio1 = self._width_mult_list[ratio[1].argmax()]
            r_score1 = ratio[1][ratio[1].argmax()]
        else:
            ratio1 = ratio[1]
            r_score1 = 1.


        self.set_prun_ratio((ratio0, ratio1))

        for w, op in zip(fai, self._ops):
            flops, size_out = op.forward_flops(size)
            result = result + flops * w * r_score0 * r_score1
        return result, size_out

class Cell(nn.Module):
    def __init__(self, C_in, C_out=None, down=True, width_mult_list=[1.]):
        super(Cell, self).__init__()
        self._C_in = C_in
        if C_out is None: C_out = C_in
        self._C_out = C_out
        self._down = down
        self._width_mult_list = width_mult_list

        self._op = MixedOp(C_in, C_out, width_mult_list=width_mult_list)

        if self._down:
            self.downsample = MixedOp(C_in, C_in*2, stride=2, width_mult_list=width_mult_list)

    def forward(self, input, fais, thetas):
        # thetas: (in, out, down)
        out = self._op(input, fais, (thetas[0], thetas[1]))
        assert (self._down and (thetas[2] is not None)) or ((not self._down) and (thetas[2] is None))
        down = self.downsample(input, fais, (thetas[0], thetas[2])) if self._down else None
        return out, down

    def forward_latency(self, size, fais, thetas):
        # thetas: (in, out, down)
        out = self._op.forward_latency(size, fais, (thetas[0], thetas[1]))
        assert (self._down and (thetas[2] is not None)) or ((not self._down) and (thetas[2] is None))
        down = self.downsample.forward_latency(size, fais, (thetas[0], thetas[2])) if self._down else None
        return out, down

    def forward_flops(self, size, fais, thetas):
        # thetas: (in, out, down)
        out = self._op.forward_flops(size, fais, (thetas[0], thetas[1]))
        assert (self._down and (thetas[2] is not None)) or ((not self._down) and (thetas[2] is None))
        down = self.downsample.forward_latency(size, fais, (thetas[0], thetas[2])) if self._down else None
        return out, down

class get_disp(nn.Module):
    def __init__(self, num_in_layers):
        super(get_disp, self).__init__()
        self.conv1 = nn.Conv2d(num_in_layers, 2, kernel_size=3, stride=1)
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        p = 1
        p2d = (p, p, p, p)
        x = self.conv1(F.pad(x, p2d))
        x = self.normalize(x)
        return 0.3 * self.sigmoid(x)

def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    if(img.size()==3):
       img = torch.randn(1,img.size(0),img.size(1),img.size(2))
    img = img.cuda()
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        self.ssim = SSIM()

        target = torch.randn(self.batch_size, target.size(0), target.size(1), target.size(2))
        target = target.cuda()
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        if(pred.dim() == 3):
            pred = torch.randn(1,pred.size(0),pred.size(1),pred.size(2))
            pred = pred.cuda()
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


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
        smooth_loss = get_smooth_loss(self,norm_disp, color)

        loss += self.disparity_smoothness * smooth_loss / (2 ** scale)
        total_loss += loss
        losses["loss/{}".format(scale)] = loss

    total_loss /= self.nums_scales
    losses["loss"] = total_loss
    return losses

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

def generate_images_pred(self, inputs, outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    self.backproject_depth = {}
    self.project_3d = {}
    for scale in self.scales:
        h = self.height // (2 ** scale)
        w = self.width // (2 ** scale)

        self.backproject_depth[scale] = BackprojectDepth(self.batch_size, h, w)
        self.backproject_depth[scale] = self.backproject_depth[scale].cuda()

        self.project_3d[scale] = Project3D(self.batch_size, h, w)
        self.project_3d[scale] = self.project_3d[scale].cuda()

    for scale in self.scales:
        disp = outputs[("disp", scale)]
        if self.v1_multiscale:
            source_scale = scale
        else:
            disp = F.interpolate(disp, [self.height, self.width], mode="bilinear", align_corners=False)
            source_scale = 0

        _, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

        outputs[("depth", 0, scale)] = depth

        for i, frame_id in enumerate(self.frame_ids[1:]):

            if frame_id == "s":
                T = inputs["stereo_T"]
            else:
                T = outputs[("cam_T_cam", 0, frame_id)]

            # from the authors of https://arxiv.org/abs/1712.00175
            if self.pose_model_type == "posecnn":

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
            if(inputs[("color", frame_id, source_scale)].dim()==3):
              inputs[("color", frame_id, source_scale)] = torch.randn(self.batch_size,inputs[("color", frame_id, source_scale)].size(0),inputs[("color", frame_id, source_scale)].size(1),inputs[("color", frame_id, source_scale)].size(2))
            inputs[("color", frame_id, source_scale)] = inputs[("color", frame_id, source_scale)].cuda()
            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")
            inputs[("color", frame_id, source_scale)] = torch.randn(inputs[("color", frame_id, source_scale)].size(1),inputs[("color", frame_id, source_scale)].size(2),inputs[("color", frame_id, source_scale)].size(3))
            inputs[("color", frame_id, source_scale)] = inputs[("color", frame_id, source_scale)].cuda()

            if not self.disable_automasking:
                outputs[("color_identity", frame_id, scale)] = \
                    inputs[("color", frame_id, source_scale)]
from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        x = x.cuda()
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs

class BaseDecodeHead(nn.Module):
    """Base class for BaseDecodeHead.

    Args:
        in_channels (int|Sequence[int]): Input channels.
        channels (int): Channels after modules, before conv_seg.
        num_classes (int): Number of classes.
        dropout_ratio (float): Ratio of dropout layer. Default: 0.1.
        conv_cfg (dict|None): Config of conv layers. Default: None.
        norm_cfg (dict|None): Config of norm layers. Default: None.
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
        in_index (int|Sequence[int]): Input feature index. Default: -1
        input_transform (str|None): Transformation type of input features.
            Options: 'resize_concat', 'multiple_select', None.
            'resize_concat': Multiple feature maps will be resize to the
                same size as first one and than concat together.
                Usually used in FCN head of HRNet.
            'multiple_select': Multiple feature maps will be bundle into
                a list and passed into decode head.
            None: Only one select feature map is allowed.
            Default: None.
        loss_decode (dict): Config of decode loss.
            Default: dict(type='CrossEntropyLoss').
        ignore_index (int | None): The label index to be ignored. When using
            masked BCE loss, ignore_index should be set to None. Default: 255
        sampler (dict|None): The config of segmentation map sampler.
            Default: None.
        align_corners (bool): align_corners argument of F.interpolate.
            Default: False.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index=-1,
                 input_transform=None,
                 loss_decode=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self._init_inputs(in_channels, in_index, input_transform)
        self.channels = channels
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index
        # self.loss_decode = build_loss(loss_decode)
        self.ignore_index = ignore_index
        self.align_corners = align_corners
        # if sampler is not None:
        #     self.sampler = build_pixel_sampler(sampler, context=self)
        # else:
        #     self.sampler = None

        # self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        # self.conv1 = nn.Conv2d(channels, num_classes, 3, padding=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def init_weights(self):
        """Initialize weights of classification layer."""
        # normal_init(self.conv_seg, mean=0, std=0.01)
        # normal_init(self.conv1, mean=0, std=0.01)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        seg_logits = self.forward(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses

    def forward_test(self, inputs, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs)

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            # == if batch size = 1, BN is not supported, change to GN
            if pool_scale == 1: norm_cfg = dict(type='GN', requires_grad=True, num_groups=256)
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=self.act_cfg)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


class PSP(BaseDecodeHead):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(PSP, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        """Forward function."""
        inputs = self._transform_inputs(inputs)

        return self.psp_forward(inputs)



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, v_dim, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(v_dim, v_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, v, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qk = self.qk(x).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # assert self.dim % v.shape[-1] == 0, "self.dim % v.shape[-1] != 0"
        # repeat_num = self.dim // v.shape[-1]
        # v = v.view(B_, N, self.num_heads // repeat_num, -1).transpose(1, 2).repeat(1, repeat_num, 1, 1)

        assert self.dim == v.shape[-1], "self.dim != v.shape[-1]"
        v = v.view(B_, N, self.num_heads, -1).transpose(1, 2)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
# From PyTorch internals

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CRFBlock(nn.Module):
    """ CRF Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, v_dim, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.v_dim = v_dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, v_dim=v_dim,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(v_dim)
        mlp_hidden_dim = int(v_dim * mlp_ratio)
        self.mlp = Mlp(in_features=v_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, v, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift.
        """
        B, L, C = x.shape
        H, W = self.H, self.W
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        v = F.pad(v, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_v = torch.roll(v, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            shifted_v = v
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(shifted_v, self.window_size)  # nW*B, window_size, window_size, C
        v_windows = v_windows.view(-1, self.window_size * self.window_size,
                                   v_windows.shape[-1])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, v_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.v_dim)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, self.v_dim)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicCRFLayer(nn.Module):
    """ A basic NeWCRFs layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 v_dim,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CRFBlock(
                dim=dim,
                num_heads=num_heads,
                v_dim=v_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, v, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            blk.H, blk.W = H, W
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, v, attn_mask)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class NewCRF(nn.Module):
    def __init__(self,
                 input_dim=96,
                 embed_dim=96,
                 v_dim=64,
                 window_size=7,
                 num_heads=4,
                 depth=2,
                 patch_size=4,
                 in_chans=3,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        if input_dim != embed_dim:
            self.proj_x = nn.Conv2d(input_dim, embed_dim, 3, padding=1)
        else:
            self.proj_x = None

        if v_dim != embed_dim:
            self.proj_v = nn.Conv2d(v_dim, embed_dim, 3, padding=1)
        elif embed_dim % v_dim == 0:
            self.proj_v = None

        # For now, v_dim need to be equal to embed_dim, because the output of window-attn is the input of shift-window-attn
        v_dim = embed_dim
        assert v_dim == embed_dim

        self.crf_layer = BasicCRFLayer(
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            v_dim=v_dim,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=False)

        layer = norm_layer(embed_dim)
        layer_name = 'norm_crf'
        self.add_module(layer_name, layer)

    def forward(self, x, v):
        if self.proj_x is not None:
            x = self.proj_x(x)
        if self.proj_v is not None:
            v = self.proj_v(v)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        v = v.transpose(1, 2).transpose(2, 3)

        x_out, H, W, x, Wh, Ww = self.crf_layer(x, v, Wh, Ww)
        norm_layer = getattr(self, f'norm_crf')
        x_out = norm_layer(x_out)
        out = x_out.view(-1, H, W, self.embed_dim).permute(0, 3, 1, 2).contiguous()

        return out

class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class Network_Multi_Path(nn.Module):
    def __init__(self, num_classes=19, layers=16, criterion=nn.CrossEntropyLoss(ignore_index=-1), Fch=12, width_mult_list=[1.,], prun_modes=['arch_ratio',], stem_head_width=[(1., 1.),]):
        super(Network_Multi_Path, self).__init__()
        self._num_classes = num_classes
        assert layers >= 3
        self._layers = layers
        self._criterion = criterion
        self._Fch = Fch
        self._width_mult_list = width_mult_list
        self._prun_modes = prun_modes
        self.prun_mode = None # prun_mode is higher priority than _prun_modes
        self._stem_head_width = stem_head_width
        self._flops = 0
        self._params = 0
        #self.lossdepth =
        """
            stem由5个3*3的卷积组成
            """
        self.stem = nn.ModuleList([
            nn.Sequential(
                ConvNorm(3, self.num_filters(2, stem_ratio)*2, kernel_size=3, stride=2, padding=1, bias=False, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(2, stem_ratio)*2, self.num_filters(4, stem_ratio)*2, kernel_size=3, stride=2, groups=1, slimmable=False),
                BasicResidual2x(self.num_filters(4, stem_ratio)*2, self.num_filters(8, stem_ratio), kernel_size=3, stride=2, groups=1, slimmable=False)
            ) for stem_ratio, _ in self._stem_head_width ])
        #构建基础Cell
        #########depth###############################
        self.convdepth1 = resblock_basic(3,self.num_filters(2, 1)*2,2,2)
        self.convdepth2 = resblock_basic(self.num_filters(2, 1)*2, 192,2,2)
        self.convd = conv()
        self.cells = nn.ModuleList()
        for l in range(layers):# 网络层数
            cells = nn.ModuleList()
            if l == 0:
                # first node has only one input (prev cell's output)
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
            elif l == 1:#第二层
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list))
            elif l < layers - 1:#中间层
                cells.append(Cell(self.num_filters(8), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list))
            else:#最后一层
                cells.append(Cell(self.num_filters(8), down=False, width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(16), down=False, width_mult_list=width_mult_list))
                cells.append(Cell(self.num_filters(32), down=False, width_mult_list=width_mult_list))
            self.cells.append(cells)

        self.refine32 = nn.ModuleList([
            nn.ModuleList([
                ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(32, head_ratio), self.num_filters(16, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width ])
        self.refine16 = nn.ModuleList([
            nn.ModuleList([
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=1, bias=False, groups=1, slimmable=False),
                ConvNorm(self.num_filters(16, head_ratio), self.num_filters(8, head_ratio), kernel_size=3, padding=1, bias=False, groups=1, slimmable=False)]) for _, head_ratio in self._stem_head_width ])

        self.head0 = nn.ModuleList([ Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head1 = nn.ModuleList([ Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head2 = nn.ModuleList([ Head(self.num_filters(8, head_ratio), num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head02 = nn.ModuleList([ Head(self.num_filters(8, head_ratio)*2, num_classes, False) for _, head_ratio in self._stem_head_width ])
        self.head12 = nn.ModuleList([ Head(self.num_filters(8, head_ratio)*2, num_classes, False) for _, head_ratio in self._stem_head_width ])

        # contains arch_param names: {"fais": fais, "mjus": mjus, "thetas": thetas}
        self._arch_names = []
        self._arch_parameters = []
        for i in range(len(self._prun_modes)):
            arch_name, arch_param = self._build_arch_parameters(i)
            self._arch_names.append(arch_name)
            self._arch_parameters.append(arch_param)
            self._reset_arch_parameters(i)
        # switch set of arch if we have more than 1 arch
        self.arch_idx = 0

        self.conv_3x3 = ConvBnRelu(192, 192, 3, 1, 1, has_bn=False, norm_layer=1, has_relu=True,
                          has_bias=False).cuda()
        self.conv_1x1 = nn.Conv2d(96, 384, kernel_size=1, stride=1, padding=0).cuda()
        self.conv_3x3_16 = ConvBnRelu(384, 384, 3, 1, 1, has_bn=False, norm_layer=1, has_relu=True,
                                 has_bias=False).cuda()
        self.conv_1x1_16 = nn.Conv2d(96, 768, kernel_size=1, stride=1, padding=0).cuda()

        self.conv_3x3_32 = ConvBnRelu(384, 768, 3, 1, 1, has_bn=False, norm_layer=1, has_relu=True,
                                 has_bias=False).cuda()
        self.conv_1x1_32 = nn.Conv2d(192, 1536, kernel_size=1, stride=1, padding=0).cuda()

        in_channels = [192, 384, 768, 1536]
        norm_cfg = dict(type='BN', requires_grad=True)
        embed_dim = 512
        decoder_cfg = dict(
            in_channels=in_channels,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=embed_dim,
            dropout_ratio=0.0,
            num_classes=32,
            norm_cfg=norm_cfg,
            align_corners=False
        )

        self.decoder = PSP(**decoder_cfg).cuda()

        v_dim = decoder_cfg['num_classes']*4
        win = 7
        crf_dims = [128, 256, 512, 1024]
        v_dims = [64, 128, 256, embed_dim]
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3],
                      num_heads=32).cuda()
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2],
                      num_heads=16).cuda()
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1],
                      num_heads=8).cuda()
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0],
                      num_heads=4).cuda()
        self.disp_head1 = DispHead(input_dim=128).cuda()

    def num_filters(self, scale, width=1.0):
        return int(np.round(scale * self._Fch * width))

    def new(self):
        model_new = Network(self._num_classes, self._layers, self._criterion, self._Fch).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
                x.data.copy_(y.data)
        return model_new

    def sample_prun_ratio(self, mode="arch_ratio"):
        '''
        mode: "min"|"max"|"random"|"arch_ratio"(default)
        '''
        assert mode in ["min", "max", "random", "arch_ratio"]
        if mode == "arch_ratio":
            thetas = self._arch_names[0]["thetas"]
            thetas0 = getattr(self, thetas[0])
            thetas0_sampled = []
            for layer in range(self._layers - 1):
                thetas0_sampled.append(gumbel_softmax(F.log_softmax(thetas0[layer], dim=-1), hard=True))
            thetas1 = getattr(self, thetas[1])
            thetas1_sampled = []
            for layer in range(self._layers - 1):
                thetas1_sampled.append(gumbel_softmax(F.log_softmax(thetas1[layer], dim=-1), hard=True))
            thetas2 = getattr(self, thetas[2])
            thetas2_sampled = []
            for layer in range(self._layers - 2):
                thetas2_sampled.append(gumbel_softmax(F.log_softmax(thetas2[layer], dim=-1), hard=True))
            return [thetas0_sampled, thetas1_sampled, thetas2_sampled]
        elif mode == "min":
            thetas0_sampled = []
            for layer in range(self._layers - 1):
                thetas0_sampled.append(self._width_mult_list[0])
            thetas1_sampled = []
            for layer in range(self._layers - 1):
                thetas1_sampled.append(self._width_mult_list[0])
            thetas2_sampled = []
            for layer in range(self._layers - 2):
                thetas2_sampled.append(self._width_mult_list[0])
            return [thetas0_sampled, thetas1_sampled, thetas2_sampled]
        elif mode == "max":
            thetas0_sampled = []
            for layer in range(self._layers - 1):
                thetas0_sampled.append(self._width_mult_list[-1])
            thetas1_sampled = []
            for layer in range(self._layers - 1):
                thetas1_sampled.append(self._width_mult_list[-1])
            thetas2_sampled = []
            for layer in range(self._layers - 2):
                thetas2_sampled.append(self._width_mult_list[-1])
            return [thetas0_sampled, thetas1_sampled, thetas2_sampled]
        elif mode == "random":
            thetas0_sampled = []
            for layer in range(self._layers - 1):
                thetas0_sampled.append(np.random.choice(self._width_mult_list))
            thetas1_sampled = []
            for layer in range(self._layers - 1):
                thetas1_sampled.append(np.random.choice(self._width_mult_list))
            thetas2_sampled = []
            for layer in range(self._layers - 2):
                thetas2_sampled.append(np.random.choice(self._width_mult_list))
            return [thetas0_sampled, thetas1_sampled, thetas2_sampled]


    def forward(self, arg, input,inputs,eval,test):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        refine16 = self.refine16[0]
        refine32 = self.refine32[0]
        head0 = self.head0[0]
        head1 = self.head1[0]
        head2 = self.head2[0]
        head02 = self.head02[0]
        head12 = self.head12[0]
        if(eval==False):
           input = torch.randn((arg.batch_size,3,352,1120))
        input = input.cuda(non_blocking=True)
        x1 = self.convdepth1(input)
        x2 = self.convdepth2(x1)

        stem = self.stem[0]

        fais0 = F.softmax(getattr(self, self._arch_names[0]["fais"][0]), dim=-1).cuda()
        fais1 = F.softmax(getattr(self, self._arch_names[0]["fais"][1]), dim=-1).cuda()
        fais2 = F.softmax(getattr(self, self._arch_names[0]["fais"][2]), dim=-1).cuda()
        fais = [fais0, fais1, fais2]
        mjus1 = F.softmax(getattr(self, self._arch_names[0]["mjus"][0]), dim=-1).cuda()
        mjus2 = F.softmax(getattr(self, self._arch_names[0]["mjus"][1]), dim=-1).cuda()
        mjus = [None, mjus1, mjus2]
        if self.prun_mode is not None:
            thetas = self.sample_prun_ratio(mode=self.prun_mode)
        else:
            thetas = self.sample_prun_ratio(mode=self._prun_modes[0])

        out_prev = [[stem(input), None]]  # stem: one cell


        #out_prev = [[stem(input), None]] # stem: one cell
        # i: layer | j: scale
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                fai = fais[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[0][0], thetas[j][i-j], thetas[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (thetas[j][i-j-1], self._stem_head_width[0][1], None)
                    else:
                        ratio = (thetas[j][i-j], self._stem_head_width[0][1], None)
                elif j == 2:
                    # cell in last scale: no down ratio "None"
                    ratio = (thetas[j][i-j], thetas[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (thetas[j][i-j-1], thetas[j][i-j], thetas[j+1][i-j])
                    else:
                        ratio = (thetas[j][i-j], thetas[j][i-j+1], thetas[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell(out_prev[0][0], fai, ratio)
                    out.append((out1, down1))
                else:
                    if i == j:
                        out0, down0 = cell(out_prev[j-1][1], fai, ratio)
                        out.append((out0, down0))
                    else:
                        if mjus[j][i-j-1][0] > 0:
                            out0, down0 = cell(out_prev[j-1][1], fai, ratio)
                        if mjus[j][i-j-1][1] > 0:
                            out1, down1 = cell(out_prev[j][0], fai, ratio)
                        out.append((
                            sum(w * out for w, out in zip(mjus[j][i-j-1], [out0, out1])),
                            sum(w * down if down is not None else 0 for w, down in zip(mjus[j][i-j-1], [down0, down1])),
                            ))
            out_prev = out
        ###################################
        out0 = None; out1 = None; out2 = None
        #pose
        outputs = {}
        features = []
        output = {}
        out0 = out[0][0]
        out8 = self.conv_1x1(out0)
        out1 = refine16[0](out[1][0])
        out16 = self.conv_1x1_16(out1)
        out2 = refine32[0](out[2][0])
        out32 = self.conv_1x1_32(out2)
        x3 = out8  # 64*128
        x4 = out16  # 32*64
        x5 = out32  # 16*32
        #features.append(x1)
        features.append(x2)
        features.append(x3)
        features.append(x4)
        features.append(x5)
        ###decode##################

        ppm_out = self.decoder(features)
        e3 = self.crf3(features[3], ppm_out)
        e3 = nn.PixelShuffle(2)(e3)
        e2 = self.crf2(features[2], e3)
        e2 = nn.PixelShuffle(2)(e2)
        e1 = self.crf1(features[1], e2)
        e1 = nn.PixelShuffle(2)(e1)
        e0 = self.crf0(features[0], e1)

        d1 = self.disp_head1(e0, 4)
        depth = d1 * 40.0
        return depth


        ###################################

    def forward_latency(self, size, fai=True, beta=True, ratio=True):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        stem = self.stem[0]

        if fai:
            fais0 = F.softmax(getattr(self, self._arch_names[0]["fais"][0]), dim=-1)
            fais1 = F.softmax(getattr(self, self._arch_names[0]["fais"][1]), dim=-1)
            fais2 = F.softmax(getattr(self, self._arch_names[0]["fais"][2]), dim=-1)
            fais = [fais0, fais1, fais2]
        else:
            fais = [
                torch.ones_like(getattr(self, self._arch_names[0]["fais"][0])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[0]["fais"][1])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[0]["fais"][2])).cuda() * 1./len(PRIMITIVES)]
        if beta:
            mjus1 = F.softmax(getattr(self, self._arch_names[0]["mjus"][0]), dim=-1)
            mjus2 = F.softmax(getattr(self, self._arch_names[0]["mjus"][1]), dim=-1)
            mjus = [None, mjus1, mjus2]
        else:
            mjus = [
                None,
                torch.ones_like(getattr(self, self._arch_names[0]["mjus"][0])).cuda() * 1./2,
                torch.ones_like(getattr(self, self._arch_names[0]["mjus"][1])).cuda() * 1./2]
        if ratio:
            # thetas = self.sample_prun_ratio(mode='arch_ratio')
            if self.prun_mode is not None:
                thetas = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                thetas = self.sample_prun_ratio(mode=self._prun_modes[0])
        else:
            thetas = self.sample_prun_ratio(mode='max')

        stem_latency = 0
        latency, size = stem[0].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[1].forward_latency(size); stem_latency = stem_latency + latency
        latency, size = stem[2].forward_latency(size); stem_latency = stem_latency + latency
        out_prev = [[size, None]] # stem: one cell
        latency_total = [[stem_latency, 0], [0, 0], [0, 0]] # (out, down)

        # i: layer | j: scale
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            latency = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                fai = fais[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[0][0], thetas[j][i-j], thetas[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (thetas[j][i-j-1], self._stem_head_width[0][1], None)
                    else:
                        ratio = (thetas[j][i-j], self._stem_head_width[0][1], None)
                elif j == 2:
                    # cell in last scale
                    ratio = (thetas[j][i-j], thetas[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (thetas[j][i-j-1], thetas[j][i-j], thetas[j+1][i-j])
                    else:
                        ratio = (thetas[j][i-j], thetas[j][i-j+1], thetas[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell.forward_latency(out_prev[0][0], fai, ratio)
                    out.append((out1[1], down1[1] if down1 is not None else None))
                    latency.append([out1[0], down1[0] if down1 is not None else None])
                else:
                    if i == j:
                        out0, down0 = cell.forward_latency(out_prev[j-1][1], fai, ratio)
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        latency.append([out0[0], down0[0] if down0 is not None else None])
                    else:
                        if mjus[j][i-j-1][0] > 0:
                            # from down
                            out0, down0 = cell.forward_latency(out_prev[j-1][1], fai, ratio)
                        if mjus[j][i-j-1][1] > 0:
                            # from keep
                            out1, down1 = cell.forward_latency(out_prev[j][0], fai, ratio)
                        assert (out0 is None and out1 is None) or out0[1] == out1[1]
                        assert (down0 is None and down1 is None) or down0[1] == down1[1]
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        latency.append([
                            sum(w * out for w, out in zip(mjus[j][i-j-1], [out0[0], out1[0]])),
                            sum(w * down if down is not None else 0 for w, down in zip(mjus[j][i-j-1], [down0[0] if down0 is not None else None, down1[0] if down1 is not None else None])),
                        ])
            out_prev = out
            for ii, lat in enumerate(latency):
                # layer: i | scale: ii
                if ii == 0:
                    # only from keep
                    if lat[0] is not None: latency_total[ii][0] = latency_total[ii][0] + lat[0]
                    if lat[1] is not None: latency_total[ii][1] = latency_total[ii][0] + lat[1]
                else:
                    if i == ii:
                        # only from down
                        if lat[0] is not None: latency_total[ii][0] = latency_total[ii-1][1] + lat[0]
                        if lat[1] is not None: latency_total[ii][1] = latency_total[ii-1][1] + lat[1]
                    else:
                        if lat[0] is not None: latency_total[ii][0] = mjus[j][i-j-1][1] * latency_total[ii][0] + mjus[j][i-j-1][0] * latency_total[ii-1][1] + lat[0]
                        if lat[1] is not None: latency_total[ii][1] = mjus[j][i-j-1][1] * latency_total[ii][0] + mjus[j][i-j-1][0] * latency_total[ii-1][1] + lat[1]
        ###################################
        latency0 = latency_total[0][0]
        latency1 = latency_total[1][0]
        latency2 = latency_total[2][0]
        latency = sum([latency0, latency1, latency2])
        return latency
        ###################################

    def forward_flops(self, size, fai=True, beta=True, ratio=True):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        stem = self.stem[0]

        if fai:
            fais0 = F.softmax(getattr(self, self._arch_names[0]["fais"][0]), dim=-1)
            fais1 = F.softmax(getattr(self, self._arch_names[0]["fais"][1]), dim=-1)
            fais2 = F.softmax(getattr(self, self._arch_names[0]["fais"][2]), dim=-1)
            fais = [fais0, fais1, fais2]
        else:
            fais = [
                torch.ones_like(getattr(self, self._arch_names[0]["fais"][0])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[0]["fais"][1])).cuda() * 1./len(PRIMITIVES),
                torch.ones_like(getattr(self, self._arch_names[0]["fais"][2])).cuda() * 1./len(PRIMITIVES)]
        if beta:
            mjus1 = F.softmax(getattr(self, self._arch_names[0]["mjus"][0]), dim=-1)
            mjus2 = F.softmax(getattr(self, self._arch_names[0]["mjus"][1]), dim=-1)
            mjus = [None, mjus1, mjus2]
        else:
            mjus = [
                None,
                torch.ones_like(getattr(self, self._arch_names[0]["mjus"][0])).cuda() * 1./2,
                torch.ones_like(getattr(self, self._arch_names[0]["mjus"][1])).cuda() * 1./2]
        if ratio:
            # thetas = self.sample_prun_ratio(mode='arch_ratio')
            if self.prun_mode is not None:
                thetas = self.sample_prun_ratio(mode=self.prun_mode)
            else:
                thetas = self.sample_prun_ratio(mode=self._prun_modes[0])
        else:
            thetas = self.sample_prun_ratio(mode='max')

        stem_flops = 0
        flops, size = stem[0].forward_flops(size); stem_flops = stem_flops + flops
        flops, size = stem[1].forward_flops(size); stem_flops = stem_flops + flops
        flops, size = stem[2].forward_flops(size); stem_flops = stem_flops + flops
        out_prev = [[size, None]] # stem: one cell
        flops_total = [[stem_flops, 0], [0, 0], [0, 0]] # (out, down)

        # i: layer | j: scale
        for i, cells in enumerate(self.cells):
            # layers
            out = []
            flops = []
            for j, cell in enumerate(cells):
                # scales
                # out,down -- 0: from down; 1: from keep
                out0 = None; out1 = None
                down0 = None; down1 = None
                fai = fais[j][i-j]
                # ratio: (in, out, down)
                # int: force #channel; tensor: arch_ratio; float(<=1): force width
                if i == 0 and j == 0:
                    # first cell
                    ratio = (self._stem_head_width[0][0], thetas[j][i-j], thetas[j+1][i-j])
                elif i == self._layers - 1:
                    # cell in last layer
                    if j == 0:
                        ratio = (thetas[j][i-j-1], self._stem_head_width[0][1], None)
                    else:
                        ratio = (thetas[j][i-j], self._stem_head_width[0][1], None)
                elif j == 2:
                    # cell in last scale
                    ratio = (thetas[j][i-j], thetas[j][i-j+1], None)
                else:
                    if j == 0:
                        ratio = (thetas[j][i-j-1], thetas[j][i-j], thetas[j+1][i-j])
                    else:
                        ratio = (thetas[j][i-j], thetas[j][i-j+1], thetas[j+1][i-j])
                # out,down -- 0: from down; 1: from keep
                if j == 0:
                    out1, down1 = cell.forward_flops(out_prev[0][0], fai, ratio)
                    out.append((out1[1], down1[1] if down1 is not None else None))
                    flops.append([out1[0], down1[0] if down1 is not None else None])
                else:
                    if i == j:
                        out0, down0 = cell.forward_flops(out_prev[j-1][1], fai, ratio)
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        flops.append([out0[0], down0[0] if down0 is not None else None])
                    else:
                        if mjus[j][i-j-1][0] > 0:
                            # from down
                            out0, down0 = cell.forward_flops(out_prev[j-1][1], fai, ratio)
                        if mjus[j][i-j-1][1] > 0:
                            # from keep
                            out1, down1 = cell.forward_flops(out_prev[j][0], fai, ratio)
                        assert (out0 is None and out1 is None) or out0[1] == out1[1]
                        assert (down0 is None and down1 is None) or down0[1] == down1[1]
                        out.append((out0[1], down0[1] if down0 is not None else None))
                        flops.append([
                            sum(w * out for w, out in zip(mjus[j][i-j-1], [out0[0], out1[0]])),
                            sum(w * down if down is not None else 0 for w, down in zip(mjus[j][i-j-1], [down0[0] if down0 is not None else None, down1[0] if down1 is not None else None])),
                        ])
            out_prev = out
            for ii, lat in enumerate(flops):
                # layer: i | scale: ii
                if ii == 0:
                    # only from keep
                    if lat[0] is not None: flops_total[ii][0] = flops_total[ii][0] + lat[0]
                    if lat[1] is not None: flops_total[ii][1] = flops_total[ii][0] + lat[1]
                else:
                    if i == ii:
                        # only from down
                        if lat[0] is not None: flops_total[ii][0] = flops_total[ii-1][1] + lat[0]
                        if lat[1] is not None: flops_total[ii][1] = flops_total[ii-1][1] + lat[1]
                    else:
                        if lat[0] is not None: flops_total[ii][0] = mjus[j][i-j-1][1] * flops_total[ii][0] + mjus[j][i-j-1][0] * flops_total[ii-1][1] + lat[0]
                        if lat[1] is not None: flops_total[ii][1] = mjus[j][i-j-1][1] * flops_total[ii][0] + mjus[j][i-j-1][0] * flops_total[ii-1][1] + lat[1]
        ###################################
        flops0 = flops_total[0][0]
        flops1 = flops_total[1][0]
        flops2 = flops_total[2][0]
        flops = sum([flops0, flops1, flops2])
        return flops
        ###################################
    def _loss(self, arg, input, target, pretrain=False):
        losses = []
        val_losses = []
        running_val_loss = 0.0
        if pretrain is not True:
            # "random width": sampled by gambel softmax
            self.prun_mode = None
            for idx in range(len(self._arch_names)):
                #self.arch_idx = idx
                depth_est = self(arg,input,target,False,False)
                depth_gt = torch.autograd.Variable(target['depth'].cuda())
                mask = depth_gt > 1.0
                silog_criterion = silog_loss(variance_focus=arg.variance_focus).cuda()
                loss = silog_criterion.forward(arg,depth_est, depth_gt, mask.to(torch.bool))
                loss.backward()
                #loss = loss + sum(self._criterion(logit, target) for logit in logits)
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            depth_est = self(arg, input, target, False,False)
            depth_gt = torch.autograd.Variable(target['depth'].cuda())
            mask = depth_gt > 1.0
            silog_criterion = silog_loss(variance_focus=arg.variance_focus).cuda()
            loss = silog_criterion.forward(arg, depth_est, depth_gt, mask.to(torch.bool))
            loss.backward()
        return loss

    def _build_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)

        # define names
        fais = [ "fai_"+str(idx)+"_"+str(scale) for scale in [0, 1, 2] ]
        mjus = [ "beta_"+str(idx)+"_"+str(scale) for scale in [1, 2] ]

        setattr(self, fais[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers, num_ops), requires_grad=True)))
        setattr(self, fais[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_ops), requires_grad=True)))
        setattr(self, fais[2], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, num_ops), requires_grad=True)))
        # mjus are now in-degree probs
        # 0: from down; 1: from keep
        setattr(self, mjus[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, 2), requires_grad=True)))
        setattr(self, mjus[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-3, 2), requires_grad=True)))

        thetas = [ "ratio_"+str(idx)+"_"+str(scale) for scale in [0, 1, 2] ]
        if self._prun_modes[idx] == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1
        setattr(self, thetas[0], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_widths), requires_grad=True)))
        setattr(self, thetas[1], nn.Parameter(Variable(1e-3*torch.ones(self._layers-1, num_widths), requires_grad=True)))
        setattr(self, thetas[2], nn.Parameter(Variable(1e-3*torch.ones(self._layers-2, num_widths), requires_grad=True)))





        return {"fais": fais, "mjus": mjus, "thetas": thetas}, [getattr(self, name) for name in fais] + [getattr(self, name) for name in mjus] + [getattr(self, name) for name in thetas]

    def _reset_arch_parameters(self, idx):
        num_ops = len(PRIMITIVES)
        if self._prun_modes[idx] == 'arch_ratio':
            # prunning ratio
            num_widths = len(self._width_mult_list)
        else:
            num_widths = 1

        getattr(self, self._arch_names[idx]["fais"][0]).data = Variable(1e-3*torch.ones(self._layers, num_ops), requires_grad=True)
        getattr(self, self._arch_names[idx]["fais"][1]).data = Variable(1e-3*torch.ones(self._layers-1, num_ops), requires_grad=True)
        getattr(self, self._arch_names[idx]["fais"][2]).data = Variable(1e-3*torch.ones(self._layers-2, num_ops), requires_grad=True)
        getattr(self, self._arch_names[idx]["mjus"][0]).data = Variable(1e-3*torch.ones(self._layers-2, 2), requires_grad=True)
        getattr(self, self._arch_names[idx]["mjus"][1]).data = Variable(1e-3*torch.ones(self._layers-3, 2), requires_grad=True)
        getattr(self, self._arch_names[idx]["thetas"][0]).data = Variable(1e-3*torch.ones(self._layers-1, num_widths), requires_grad=True)
        getattr(self, self._arch_names[idx]["thetas"][1]).data = Variable(1e-3*torch.ones(self._layers-1, num_widths), requires_grad=True)
        getattr(self, self._arch_names[idx]["thetas"][2]).data = Variable(1e-3*torch.ones(self._layers-2, num_widths), requires_grad=True)
        #getattr(self, self._arch_names[idx]["log_latency"][0]).data = Variable(torch.zeros((1,), requires_grad=True))
        #getattr(self, self._arch_names[idx]["log_flops"][0]).data = Variable(torch.zeros((1,), requires_grad=True))
