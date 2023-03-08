import os
import numpy as np
from config import load_parser,parse_parser
from tool.Timer import Timer
from tool.load_model import load_model
from weight import load_weight_pose,load_weight_shape
from smpl_optim import optimizeShape,multi_stage_optimize
import joblib
from tqdm import tqdm
import cv2
from file_utils import select_nf
import matplotlib.pyplot as plt
from vis import render_smpl_demo

kintree=np.array([[15,13],[13,11],[11,9],[9,5],[5,10],[10,12],[12,14],[14,16],[5,4],[4,3],[3,2],[2,1],[17,1],
                  [1,18],[17,19],[18,20],[19,21],[20,22]])

class Config:
    OPT_R = False
    OPT_T = False
    OPT_POSE = False
    OPT_SHAPE = False
    OPT_HAND = False
    OPT_EXPR = False
    ROBUST_3D_ = False
    ROBUST_3D = False
    verbose = False
    model = 'smpl'
    device = None
    def __init__(self, args=None) -> None:
        if args is not None:
            self.verbose = args.verbose
            self.model = args.model
            self.ROBUST_3D_ = args.robust3d

def smpl_from_keypoints3d(body_model, kp3ds, args,verts_type=None,
    weight_shape=None, weight_pose=None):
    model_type = body_model.model_type
    params_init = body_model.init_params(nFrames=1)

    if weight_shape is None:
        weight_shape = load_weight_shape(model_type, args.opts)
    if model_type in ['smpl', 'smplh', 'smplx']:
        # when use SMPL model, optimize the shape only with first 1-14 limbs,
        # don't use (nose, neck)
        params_shape = optimizeShape(body_model, params_init, kp3ds,
            weight_loss=weight_shape, kintree=kintree,verts_type=verts_type)
    else:
        params_shape = optimizeShape(body_model, params_init, kp3ds,
            weight_loss=weight_shape, kintree=kintree,verts_type=verts_type)
    # optimize 3D pose
    cfg = Config(args)
    cfg.device = body_model.device
    cfg.model_type = model_type
    params = body_model.init_params(nFrames=kp3ds.shape[0])
    params['shapes'] = params_shape['shapes'].copy()
    if weight_pose is None:
        weight_pose = load_weight_pose(model_type, args.opts)
    # We divide this step to two functions, because we can have different initialization method
    params = multi_stage_optimize(body_model, params, kp3ds, None, None, None, weight_pose, verts_type=verts_type,cfg=cfg)
    return params

def root_algin_np(keypoints):
    root = np.expand_dims(keypoints[:,0,:],1)
    keypoints = keypoints - root
    return keypoints

def left2right(kp):
    kp[:,:,2]=-1*kp[:,:,2]
    return kp

def get_number_dir(path):
    for root, dirs, files in os.walk(path, topdown=False):
        pass
    return len(files)

if __name__ == '__main__': 
    parser = load_parser()
    parser.add_argument('--skel', action='store_true')
    parser.add_argument("--cuda",type=int,default=0)
    args = parse_parser(parser)
    
    render_test = True
    
    origin_img_path = "/mnt/disk_2/aoyang/new_imgs"
    orgin_keypoints_path = "/mnt/disk_2/aoyang/new_skeleton"
    param_output_path = "/mnt/disk_2/aoyang/new_param"
    
    for root,_,files in os.walk(orgin_keypoints_path):
        pass
    for i in range(len(files)):
        name = files[i][:-4]
        save_path = os.path.join(param_output_path,"{}.pkl".format(name))
        kp3ds=np.load(os.path.join(orgin_keypoints_path,"{}.npy").format(name))
        kp3ds=left2right(kp3ds)
        kp3ds=root_algin_np(kp3ds)
        conf=np.transpose(np.array([[1]*kp3ds.shape[1]]))
        conf=np.array([conf]*kp3ds.shape[0])
        kp3ds=np.concatenate((kp3ds,conf),axis=2)
        conf=np.transpose(np.array([[1]*kp3ds.shape[1]]))
        conf=np.array([conf]*kp3ds.shape[0])
        kp3ds=np.concatenate((kp3ds,conf),axis=2)
        if name[11]=='3':
            args.gender='female'
        else:
            args.gender='male'
        verts_type="fusion"
        weight_shape=None
        weight_pose=None
        with Timer('Loading {}, {}'.format(args.model, args.gender), not args.verbose):
            body_model = load_model(gender=args.gender, model_type=args.model)
            params = smpl_from_keypoints3d(body_model,kp3ds,args,verts_type,weight_shape,weight_pose)
        joblib.dump(params,save_path)
        if render_test:
            orgin_img_path = os.path.join(origin_img_path,name)
            assert (get_number_dir(origin_img_path) == kp3ds.shape[0],"the frames are not matched")
            camera_params={
            "S1C1":np.array([[-0.9999773, -0.00404104, -0.00539746, 35.53189],
                                        [-0.004719454, 0.9912043, 0.1322568, 2.188294],
                                        [-0.00481553, -0.1322793, 0.9912007, 21.58087],
                                        [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S2C4":np.array([[0.9999425, 0.0002365171, 0.01072514, 36.64215],
                                        [-0.001741237,0.990078,  0.140508, 2.436002],
                                        [0.01058549,0.1405185, -0.9900214, -1.076592],
                                        [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S1C6": np.array([[-0.829555, -0.008364107, 0.5583624, 38.07797],
                            [0.0007109789,0.9998712, 0.01603408,1.745723],
                            [0.5584246, -0.01369813, 0.8294423, 20.59404],
                            [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S2C1":np.array([[-0.9999773, -0.00404104, -0.00539746, 36.62015],
                            [-0.004719454,0.9912043, 0.1322568,2.394002],
                            [-0.00481553, -0.1322793, 0.9912007, 8.175407],
                            [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S3C1":np.array([[-0.9999773, -0.00404104, -0.00539746, -21.66285],
                            [-0.004719454,0.9912043, 0.1322568,1.653003],
                            [-0.00481553, -0.1322793, 0.9912007,3.525407],
                            [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S3C2":np.array([[-0.8350205, 0.03840624, -0.5488767, -23.82485],
                            [-0.004564091,0.9970431, 0.07670899,1.421002],
                            [-0.5501998, -0.0665587, 0.8323761,1.539407],
                            [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S3C4":np.array([[0.9999425, 0.0002365171, 0.01072514, -21.61685],
                            [-0.001741237,0.990078, 0.140508,2.006002],
                            [0.01058549, 0.1405185, -0.9900214,-7.916592],
                            [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S3C6":np.array([[-0.9527137, -0.01177603, 0.303641, -19.51085],
                            [-0.004747873,0.9997036, 0.02387418,1.209002],
                            [0.3038321, -0.02130362, 0.9524872,2.718408],
                            [0.00000, 0.00000, 0.00000, 1.00000]]),
            "S4C4": np.array([[0.9999425, 0.0002365171, 0.01072514, 12.53515],
                            [-0.001741237, 0.990078, 0.140508, 2.201002],
                            [0.01058549, 0.1405185, -0.9900214, -21.45259],
                            [0.00000, 0.00000, 0.00000, 1.00000]])
            }
            faces=body_model.faces
            start, end = args.start, min(args.end, kp3ds.shape[0])
            for nf in tqdm(range(start, end), desc='render'):
                camera_pose=camera_params["S1C1"]
                img=cv2.imread(os.path.join(orgin_img_path,"{:06d}.jpg".format(nf)))
                param = select_nf(params, nf - start)
                vertices = body_model(return_verts=True, return_tensor=False, **param)[0]
                root = kp3ds[:,0,:]
                vertices += root[nf]
                image=render_smpl_demo(vertices,faces,camera_pose,img,True,False)
                plt.figure()
                plt.axis('off')
                plt.imshow(image)
                if os.path.exists(save_path) == False:
                    os.makedirs(save_path)
                plt.savefig(os.path.join(save_path,"{:06d}.jpg").format(nf),dpi=600,pad_inches=0,bbox_inches='tight')
                plt.close()
    