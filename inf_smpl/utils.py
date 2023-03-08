
import torch
import numpy as np
import os
from os.path import join
from optim import LBFGS,FittingMonitor,grad_require
import cv2
from vis_base import plot_bbox, plot_keypoints, merge
from file_utils import write_keypoints3d, write_smpl, mkout, mkdir
import matplotlib.pyplot as plt
import datetime

def deepcopy_tensor(body_params):
    for key in body_params.keys():
        body_params[key] = body_params[key].clone()
    return body_params


def get_prepare_smplx(body_params, cfg, nFrames):
    zero_pose = torch.zeros((nFrames, 3), device=cfg.device)
    if not cfg.OPT_HAND and cfg.model in ['smplh', 'smplx']:
        zero_pose_hand = torch.zeros((nFrames, body_params['poses'].shape[1] - 66), device=cfg.device)
    elif cfg.OPT_HAND and not cfg.OPT_EXPR and cfg.model == 'smplx':
        zero_pose_face = torch.zeros((nFrames, body_params['poses'].shape[1] - 78), device=cfg.device)

    def pack(new_params):
        if not cfg.OPT_HAND and cfg.model in ['smplh', 'smplx']:
            new_params['poses'] = torch.cat([zero_pose, new_params['poses'][:, 3:66], zero_pose_hand], dim=1)
        else:
            new_params['poses'] = torch.cat([zero_pose, new_params['poses'][:, 3:]], dim=1)
        return new_params
    return pack

def get_interp_by_keypoints(keypoints):
    if len(keypoints.shape) == 3: # (nFrames, nJoints, 3)
        conf = keypoints[..., -1]
    elif len(keypoints.shape) == 4: # (nViews, nFrames, nJoints)
        conf = keypoints[..., -1].sum(axis=0)
    else:
        raise NotImplementedError
    not_valid_frames = np.where(conf.sum(axis=1) < 0.01)[0].tolist()
    # 遍历空白帧，选择起点和终点
    ranges = []
    if len(not_valid_frames) > 0:
        start = not_valid_frames[0]
        for i in range(1, len(not_valid_frames)):
            if not_valid_frames[i] == not_valid_frames[i-1] + 1:
                pass
            else:# 改变位置了
                end = not_valid_frames[i-1]
                ranges.append((start, end))
                start = not_valid_frames[i]
        ranges.append((start, not_valid_frames[-1]))
    def interp_func(params):
        for start, end in ranges:
            # 对每个需要插值的区间: 这里直接使用最近帧进行插值了
            left = start - 1
            right = end + 1
            for nf in range(start, end+1):
                weight = (nf - left)/(right - left)
                for key in ['Rh', 'Th', 'poses']:
                    params[key][nf] = interp(params[key][left], params[key][right], 1-weight, key=key)
        return params
    return interp_func

def interp(left_value, right_value, weight, key='poses'):
    if key == 'Rh':
        return left_value * weight + right_value * (1 - weight)
    elif key == 'Th':
        return left_value * weight + right_value * (1 - weight)
    elif key == 'poses':
        return left_value * weight + right_value * (1 - weight)

def dict_of_tensor_to_numpy(body_params):
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params

def _optimizeSMPL(body_model, body_params, prepare_funcs, postprocess_funcs,
    loss_funcs, extra_params=None,
    weight_loss={},verts_type=None, cfg=None,descrip=None):
    """ A common interface for different optimization.

    Args:
        body_model (SMPL model)
        body_params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        prepare_funcs (List): functions for prepare
        loss_funcs (Dict): functions for loss
        weight_loss (Dict): weight
        cfg (Config): Config Node controling running mode
    """
    loss_funcs = {key: val for key, val in loss_funcs.items() if key in weight_loss.keys() and weight_loss[key] > 0.}
    if cfg.verbose:
        print('Loss Functions: ')
        for key, func in loss_funcs.items():
            print('  -> {:15s}: {}'.format(key, func.__doc__))
    opt_params = get_optParams(body_params, cfg, extra_params)
    grad_require(opt_params, True)
    optimizer = LBFGS(opt_params,
        line_search_fn='strong_wolfe')
    PRINT_STEP = 100
    records = []
    def closure(debug=False):
        # 0. Prepare body parameters => new_params
        optimizer.zero_grad()
        new_params = body_params.copy()
        for func in prepare_funcs:
            new_params = func(new_params)
        # 1. Compute keypoints => kpts_est
        kpts_est = body_model(return_verts=False, return_tensor=True, return_my_joints=True,verts_type=verts_type,**new_params)
        # kp = kpts_est[0].detach().cpu().numpy()
        # visual_skeleton(kp)
        # print(descrip)
        # 2. Compute loss => loss_dict
        loss_dict = {key:func(kpts_est=kpts_est, **new_params) for key, func in loss_funcs.items()}
        # 3. Summary and log
        cnt = len(records)
        if cfg.verbose and cnt % PRINT_STEP == 0:
            print('{:-6d}: '.format(cnt) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key])
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        records.append(loss.item())
        if debug:
            return loss_dict
        loss.backward()
        return loss

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    if cfg.verbose:
        print('{:-6d}: '.format(len(records)) + ' '.join([key + ' %f'%(loss_dict[key].item()*weight_loss[key])
            for key in loss_dict.keys() if weight_loss[key]>0]))
    loss_dict = {key:val.item() for key, val in loss_dict.items()}
    # post-process the body_parameters
    for func in postprocess_funcs:
        body_params = func(body_params)
    return body_params

def get_optParams(body_params, cfg, extra_params):
    for key, val in body_params.items():
        body_params[key] = torch.Tensor(val).to(cfg.device)
    if cfg is None:
        opt_params = [body_params['Rh'], body_params['Th'], body_params['poses']]
    else:
        if extra_params is not None:
            opt_params = extra_params
        else:
            opt_params = []
        if cfg.OPT_R:
            opt_params.append(body_params['Rh'])
        if cfg.OPT_T:
            opt_params.append(body_params['Th'])
        if cfg.OPT_POSE:
            opt_params.append(body_params['poses'])
        if cfg.OPT_SHAPE:
            opt_params.append(body_params['shapes'])
        if cfg.OPT_EXPR and cfg.model == 'smplx':
            opt_params.append(body_params['expression'])
    return opt_params


class FileWriter:
    """
        This class provides:
                      |  write  | vis
        - keypoints2d |    x    |  o
        - keypoints3d |    x    |  o
        - smpl        |    x    |  o
    """

    def __init__(self, output_path, config=None, basenames=[], cfg=None) -> None:
        self.out = output_path
        keys = ['keypoints3d', 'match', 'smpl', 'skel', 'repro', 'keypoints']
        output_dict = {key: join(self.out, key) for key in keys}
        self.output_dict = output_dict

        self.basenames = basenames
        if cfg is not None:
            print(cfg, file=open(join(output_path, 'exp.yml'), 'w'))
        self.save_origin = False
        self.config = config

    def write_keypoints2d(self, ):
        pass

    def vis_keypoints2d_mv(self, images, lDetections, outname=None,
                           vis_id=True):
        mkout(outname)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                pid = det['id']
                if 'keypoints2d' in det.keys():
                    keypoints = det['keypoints2d']
                else:
                    keypoints = det['keypoints']
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        if len(images_vis) > 1:
            images_vis = merge(images_vis, resize=not self.save_origin)
        else:
            images_vis = images_vis[0]
        if outname is not None:
            # savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            # savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(outname, images_vis)
        return images_vis

    def write_keypoints3d(self, results, outname):
        write_keypoints3d(outname, results)

    def vis_keypoints3d(self, result, outname):
        # visualize the repro of keypoints3d
        import ipdb;
        ipdb.set_trace()

    def vis_smpl(self, render_data, images, cameras, outname, add_back):
        mkout(outname)
        from easymocap.visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None)
        render_results = render.render(render_data, cameras, images, add_back=add_back)
        image_vis = merge(render_results, resize=not self.save_origin)
        cv2.imwrite(outname, image_vis)
        return image_vis

    def _write_keypoints3d(self, results, nf=-1, base=None):
        os.makedirs(self.output_dict['keypoints3d'], exist_ok=True)
        if base is None:
            base = '{:06d}'.format(nf)
        savename = join(self.output_dict['keypoints3d'], '{}.json'.format(base))
        save_json(savename, results)

    def vis_detections(self, images, lDetections, nf, key='keypoints', to_img=True, vis_id=True):
        os.makedirs(self.output_dict[key], exist_ok=True)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            for det in lDetections[nv]:
                if key == 'match' and 'id_match' in det.keys():
                    pid = det['id_match']
                else:
                    pid = det['id']
                if key not in det.keys():
                    keypoints = det['keypoints']
                else:
                    keypoints = det[key]
                if 'bbox' not in det.keys():
                    bbox = get_bbox_from_pose(keypoints, img)
                else:
                    bbox = det['bbox']
                plot_bbox(img, bbox, pid=pid, vis_id=vis_id)
                plot_keypoints(img, keypoints, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        image_vis = merge(images_vis, resize=not self.save_origin)
        if to_img:
            savename = join(self.output_dict[key], '{:06d}.jpg'.format(nf))
            cv2.imwrite(savename, image_vis)
        return image_vis

    def write_smpl(self, results, outname):
        write_smpl(outname, results)

    def vis_keypoints3d(self, infos, nf, images, cameras, mode='repro'):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        # cameras: (K, R, T)
        images_vis = []
        for nv, image in enumerate(images):
            img = image.copy()
            K, R, T = cameras['K'][nv], cameras['R'][nv], cameras['T'][nv]
            P = K @ np.hstack([R, T])
            for info in infos:
                pid = info['id']
                keypoints3d = info['keypoints3d']
                # 重投影
                kcam = np.hstack([keypoints3d[:, :3], np.ones((keypoints3d.shape[0], 1))]) @ P.T
                kcam = kcam[:, :2] / kcam[:, 2:]
                k2d = np.hstack((kcam, keypoints3d[:, -1:]))
                bbox = get_bbox_from_pose(k2d, img)
                plot_bbox(img, bbox, pid=pid, vis_id=pid)
                plot_keypoints(img, k2d, pid=pid, config=self.config, use_limb_color=False, lw=2)
            images_vis.append(img)
        savename = join(out, '{:06d}.jpg'.format(nf))
        image_vis = merge(images_vis, resize=False)
        cv2.imwrite(savename, image_vis)
        return image_vis

    def _vis_smpl(self, render_data_, nf, images, cameras, mode='smpl', base=None, add_back=False, extra_mesh=[]):
        out = join(self.out, mode)
        os.makedirs(out, exist_ok=True)
        from easymocap.visualize.renderer import Renderer
        render = Renderer(height=1024, width=1024, faces=None, extra_mesh=extra_mesh)
        if isinstance(render_data_, list):  # different view have different data
            for nv, render_data in enumerate(render_data_):
                render_results = render.render(render_data, cameras, images)
                image_vis = merge(render_results, resize=not self.save_origin)
                savename = join(out, '{:06d}_{:02d}.jpg'.format(nf, nv))
                cv2.imwrite(savename, image_vis)
        else:
            render_results = render.render(render_data_, cameras, images, add_back=add_back)
            image_vis = merge(render_results, resize=not self.save_origin)
            if nf != -1:
                if base is None:
                    base = '{:06d}'.format(nf)
                savename = join(out, '{}.jpg'.format(base))
                cv2.imwrite(savename, image_vis)
            return image_vis


def select_nf(params_all, nf):
    output = {}
    for key in ['poses', 'Rh', 'Th']:
        output[key] = params_all[key][nf:nf+1, :]
    if 'expression' in params_all.keys():
        output['expression'] = params_all['expression'][nf:nf+1, :]
    if params_all['shapes'].shape[0] == 1:
        output['shapes'] = params_all['shapes']
    else:
        output['shapes'] = params_all['shapes'][nf:nf+1, :]
    return output