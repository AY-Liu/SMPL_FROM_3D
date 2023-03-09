from os.path import join
from model.smpl_layer import SMPLlayer

def load_model(gender='neutral', use_cuda=True, model_type='smpl', skel_type='body25', device=None, model_path='../smpl_model'):
    # prepare SMPL model
    # print('[Load model {}/{}]'.format(model_type, gender))
    import torch
    if device is None:
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    if model_type == 'smpl':
        if skel_type == 'body25':
            reg_path = join(model_path, 'J_regressor_body25.npy')
        elif skel_type == 'h36m':
            reg_path = join(model_path, 'J_regressor_h36m.npy')
        else:
            raise NotImplementedError
        body_model = SMPLlayer(join(model_path, 'smpl'), gender=gender, device=device,
            regressor_path=reg_path)
    elif model_type == 'smplh':
        body_model = SMPLlayer(join(model_path, 'smplh/SMPLH_MALE.pkl'), model_type='smplh', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplh.txt'))
    elif model_type == 'smplx':
        body_model = SMPLlayer(join(model_path, 'smplx/SMPLX_{}.pkl'.format(gender.upper())), model_type='smplx', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_body25_smplx.txt'))
    elif model_type == 'manol' or model_type == 'manor':
        lr = {'manol': 'LEFT', 'manor': 'RIGHT'}
        body_model = SMPLlayer(join(model_path, 'smplh/MANO_{}.pkl'.format(lr[model_type])), model_type='mano', gender=gender, device=device,
            regressor_path=join(model_path, 'J_regressor_mano_{}.txt'.format(lr[model_type])))
    else:
        body_model = None
    body_model.to(device)
    return body_model

