import pickle
import torch
import os
import numpy as np
from tqdm import tqdm
from smpl_torch_batch import SMPLModel

def create_dataset(num_samples, dataset_name, batch_size=32, theta_var=1.0, gpu_id=[0]):
    if len(gpu_id) > 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id[0])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #print(device)

    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    model = SMPLModel(device=device, model_path = 'model_24_joints.pkl')
    
    d_poses = torch.from_numpy((np.random.rand(num_samples, pose_size) - 0.5) * theta_var)\
              .type(torch.float64).to(device)
    #d_betas = torch.from_numpy((np.random.rand(num_samples, beta_size) - 0.5) * 0.2) \
    #          .type(torch.float64).to(device)
    d_betas = torch.from_numpy(np.zeros((num_samples, beta_size)))\
              .type(torch.float64).to(device)
    __trans = torch.from_numpy(np.zeros((batch_size, 3))).type(torch.float64).to(device)
    joints = []
    for i in tqdm(range(num_samples // batch_size)):
        __poses = d_poses[i*batch_size:(i+1)*batch_size]
        __betas = d_betas[i*batch_size:(i+1)*batch_size]
        
        with torch.no_grad():
            __result, __joints = model(__betas, __poses, __trans)
        joints.append(__joints)
        #outmesh_path = './samples/smpl_torch_{}.obj'
        #for i in range(result.shape[0]):
            #model.write_obj(result[i], outmesh_path.format(i))
    d_joints = torch.cat(joints, dim=0)
    
    dataset = {
        'joints': d_joints.detach().cpu().numpy(),
        'thetas': d_poses.detach().cpu().numpy(),
        #'betas': d_betas.detach().cpu().numpy()
    }
    with open(dataset_name, 'wb') as f:
        pickle.dump(dataset, f)
    

if __name__ == '__main__':
    for tv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        create_dataset(num_samples=262144, batch_size=64, theta_var=tv, 
            dataset_name='./train_dataset_24_joints_{}.pickle'.format(tv))