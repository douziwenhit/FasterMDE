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
import networks
from layers import *
from collections import OrderedDict
from loss import MonodepthLoss
###############depth#################################

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
    model_pose_encoder = networks.ResnetEncoder(self.num_layers, self.weights_init == "pretrained",num_input_images=2)

    model_pose = networks.PoseDecoder(model_pose_encoder.num_ch_enc,num_input_features=1,num_frames_to_predict_for=2)


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

            pose_inputs[0] = torch.randn((1,3,192,640))
            pose_inputs[1] = torch.randn((1,3,192,640))
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

        target = torch.randn(1, target.size(0), target.size(1), target.size(2))
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
        smooth_loss = get_smooth_loss(norm_disp, color)

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
              inputs[("color", frame_id, source_scale)] = torch.randn(1,inputs[("color", frame_id, source_scale)].size(0),inputs[("color", frame_id, source_scale)].size(1),inputs[("color", frame_id, source_scale)].size(2))
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
        self.convdepth2 = resblock_basic(self.num_filters(2, 1)*2, self.num_filters(4, 1)*2,2,2)
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


    def forward(self, arg, input,inputs):
        # out_prev: cell-state
        # index 0: keep; index 1: down
        input = torch.randn((1,3,192,640))
        input = input.cuda(non_blocking=True)
        x1 = self.convdepth1(input)
        x2 = self.convdepth2(x1)

        stem = self.stem[0]
        refine16 = self.refine16[0]
        refine32 = self.refine32[0]
        head0 = self.head0[0]
        head1 = self.head1[0]
        head2 = self.head2[0]
        head02 = self.head02[0]
        head12 = self.head12[0]

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
        out_pose = predict_poses(arg,inputs,out)
        features = []
        x3 = out[0][0]#64*128
        x4 = out[1][0]#32*64
        x5 = out[2][0]#16*32
        features.append(x1)
        features.append(x2)
        features.append(x3)
        features.append(x4)
        features.append(x5)
        num_ch_enc = np.array([x1.size(1), x2.size(1), x3.size(1), x4.size(1), x5.size(1)])
        scales = [0,1,2,3]
        decoder = DepthDecoder(num_ch_enc, scales).cuda()
        outputs = decoder(features)
        outputs.update(out_pose)
        generate_images_pred(arg,inputs, outputs)

        return outputs
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
                logits = self(arg,input,target)
                losses = compute_losses(arg,target,logits)
                losses["loss"].backward()
                #loss = loss + sum(self._criterion(logit, target) for logit in logits)
        if len(self._width_mult_list) > 1:
            self.prun_mode = "max"
            logits = self(arg, input, target)
            losses = compute_losses(arg, target, logits)
            losses["loss"].backward()
            self.prun_mode = "min"
            logits = self(arg, input, target)
            losses = compute_losses(arg, target, logits)
            losses["loss"].backward()
        return losses

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
