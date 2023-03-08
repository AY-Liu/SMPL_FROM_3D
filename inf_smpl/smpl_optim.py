import torch
import numpy as np
from tool.Timer import Timer
from loss import LossKeypoints3D,LossSmoothBodyMean,LossSmoothPoses,LossRegPoses,LossInit,LossRegPosesZero
from utils import deepcopy_tensor,get_prepare_smplx,get_interp_by_keypoints,dict_of_tensor_to_numpy,_optimizeSMPL
from optim import LBFGS,FittingMonitor,grad_require

def multi_stage_optimize(body_model, params, kp3ds, kp2ds=None, bboxes=None, Pall=None, weight={},verts_type=None, cfg=None):
    with Timer('Optimize global RT'):
        cfg.OPT_R = True
        cfg.OPT_T = True
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, verts_type=verts_type,cfg=cfg,descrip="RT")
        # params = optimizePose(body_model, params, kp3ds, weight_loss=weight, kintree=config['kintree'], cfg=cfg)
    with Timer('Optimize 3D Pose/{} frames'.format(kp3ds.shape[0])):
        cfg.OPT_POSE = True
        cfg.ROBUST_3D = False
        params = optimizePose3D(body_model, params, kp3ds, weight=weight, verts_type=verts_type,cfg=cfg,descrip="Pose")
        if False:
            cfg.ROBUST_3D = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight,verts_type=verts_type, cfg=cfg)
        if cfg.model in ['smplh', 'smplx']:
            cfg.OPT_HAND = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight, verts_type=verts_type,cfg=cfg)
        if cfg.model == 'smplx':
            cfg.OPT_EXPR = True
            params = optimizePose3D(body_model, params, kp3ds, weight=weight,verts_type=verts_type, cfg=cfg)
    return params

def optimizePose3D(body_model, params, keypoints3d, weight, verts_type,cfg,descrip):
    """
        simple function for optimizing model pose given 3d keypoints

    Args:
        body_model (SMPL model)
        params (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints3d (nFrames, nJoints, 4): 3D keypoints
        weight (Dict): string:float
        cfg (Config): Config Node controling running mode
    """
    nFrames = keypoints3d.shape[0]
    prepare_funcs = [
        deepcopy_tensor,
        get_prepare_smplx(params, cfg, nFrames),
        get_interp_by_keypoints(keypoints3d)
    ]
    loss_funcs = {
        'k3d': LossKeypoints3D(keypoints3d, cfg).body,
        # 'smooth_body': LossSmoothBodyMean(cfg).body,
        # 'smooth_poses': LossSmoothPoses(1, nFrames, cfg).poses,
        'reg_poses': LossRegPoses(cfg).reg_body,
        'init_poses': LossInit(params, cfg).init_poses,
    }
    # if body_model.model_type != 'mano':
    #     loss_funcs['reg_poses_zero'] = LossRegPosesZero(keypoints3d, cfg).__call__
    if cfg.OPT_HAND:
        loss_funcs['k3d_hand'] = LossKeypoints3D(keypoints3d, cfg, norm='l1').hand
        loss_funcs['reg_hand'] = LossRegPoses(cfg).reg_hand
        # loss_funcs['smooth_hand'] = LossSmoothPoses(1, nFrames, cfg).hands
        loss_funcs['smooth_hand'] = LossSmoothBodyMean(cfg).hand

    if cfg.OPT_EXPR:
        loss_funcs['k3d_face'] = LossKeypoints3D(keypoints3d, cfg, norm='l1').face
        loss_funcs['reg_head'] = LossRegPoses(cfg).reg_head
        loss_funcs['reg_expr'] = LossRegPoses(cfg).reg_expr
        loss_funcs['smooth_head'] = LossSmoothPoses(1, nFrames, cfg).head

    postprocess_funcs = [
        get_interp_by_keypoints(keypoints3d),
        dict_of_tensor_to_numpy
    ]
    params = _optimizeSMPL(body_model, params, prepare_funcs, postprocess_funcs, loss_funcs, weight_loss=weight,verts_type=verts_type,cfg=cfg,descrip=descrip)
    return params

def optimizeShape(body_model, body_params, keypoints3d,
    weight_loss, kintree,verts_type):
    """ simple function for optimizing model shape given 3d keypoints

    Args:
        body_model (SMPL model)
        params_init (DictParam): poses(1, 72), shapes(1, 10), Rh(1, 3), Th(1, 3)
        keypoints (nFrames, nJoints, 3): 3D keypoints
        weight (Dict): string:float
        kintree ([[src, dst]]): list of list:int
        cfg (Config): Config Node controling running mode
    """
    device = body_model.device
    # 计算不同的骨长
    kintree = np.array(kintree, dtype=np.int)
    # limb_length: nFrames, nLimbs, 1
    limb_length = np.linalg.norm(keypoints3d[:, kintree[:, 1], :3] - keypoints3d[:, kintree[:, 0], :3], axis=2, keepdims=True)
    # conf: nFrames, nLimbs, 1
    limb_conf = np.minimum(keypoints3d[:, kintree[:, 1], 3:], keypoints3d[:, kintree[:, 0], 3:])
    limb_length = torch.Tensor(limb_length).to(device)
    limb_conf = torch.Tensor(limb_conf).to(device)
    body_params = {key:torch.Tensor(val).to(device) for key, val in body_params.items()}
    body_params_init = {key:val.clone() for key, val in body_params.items()}
    opt_params = [body_params['shapes']]
    grad_require(opt_params, True)
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe', max_iter=100,lr=1)
    nFrames = keypoints3d.shape[0]
    verbose = False
    def closure(debug=False):
        optimizer.zero_grad()
        keypoints3d = body_model(return_verts=False, return_tensor=True, only_shape=True,verts_type=verts_type,**body_params)
        # visual_skeleton(keypoints3d[0].cpu().detach())
        src = keypoints3d[:, kintree[:, 0], :3] #.detach()
        dst = keypoints3d[:, kintree[:, 1], :3]
        direct_est = (dst - src).detach()
        direct_norm = torch.norm(direct_est, dim=2, keepdim=True)
        direct_normalized = direct_est/(direct_norm + 1e-4)
        err = dst - src - direct_normalized * limb_length
        loss_dict = {
            's3d': torch.sum(err**2*limb_conf)/nFrames,
            'reg_shapes': torch.sum(body_params['shapes']**2)}
        if 'init_shape' in weight_loss.keys():
            loss_dict['init_shape'] = torch.sum((body_params['shapes'] - body_params_init['shapes'])**2)
        # fittingLog.step(loss_dict, weight_loss)
        if verbose:
            print(' '.join([key + ' %.3f'%(loss_dict[key].item()*weight_loss[key])
                for key in loss_dict.keys() if weight_loss[key]>0]))
        loss = sum([loss_dict[key]*weight_loss[key]
                    for key in loss_dict.keys()])
        if not debug:
            loss.backward()
            return loss
        else:
            return loss_dict

    fitting = FittingMonitor(ftol=1e-4)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    for key in loss_dict.keys():
        loss_dict[key] = loss_dict[key].item()
    optimizer = LBFGS(
        opt_params, line_search_fn='strong_wolfe')
    body_params = {key:val.detach().cpu().numpy() for key, val in body_params.items()}
    return body_params