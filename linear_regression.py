'''
    linear_regression.py: Try to approximate empirical solution of pose-to-theta
        with correspondence observed from interpolation results.
    
    Empirically, we guess any theta should have the form:
    
        $$ theta_i = acos(<p_j - p_k, p_l - p_k>) * w_i + b_i $$
        
    where w_i and b_i will be estimated using least square regression.
'''

from smpl_torch_batch import SMPLModel
import numpy as np
import pickle
import torch
import os
from torch.utils.data import Dataset, DataLoader

class Joint2SMPLDataset(Dataset):
    '''
        Regression Data with Joint and Theta, Beta.
        Predict Pose angles and Betas from input joints.
        Train/val: 1:1
        
        TODO: creating training testing split
    '''
    def __init__(self, pickle_file, batch_size=64,fix_beta_zero=False):
        super(Joint2SMPLDataset, self).__init__()
        assert(os.path.isfile(pickle_file))
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        
        self.thetas = dataset['thetas']
        self.joints = dataset['joints']
        self.fix_beta_zero = fix_beta_zero
        if not fix_beta_zero:
            self.betas = dataset['betas']
        
        print(self.joints.shape)
        self.batch_size = batch_size
        self.length = self.joints.shape[0]
        print(self.length)
        
    def __getitem__(self, item):
        js = self.joints[item]
        ts = self.thetas[item]
        if self.fix_beta_zero:
            bs = np.zeros(10, dtype=np.float64)
        else:
            bs = self.betas[item]
        return {'joints': js, 'thetas': ts, 'betas': bs}
        
    def rand_val_batch(self):
        length = self.length // self.batch_size
        item = np.random.randint(0, length)
        js = self.joints[item*self.batch_size: (item+1)*self.batch_size]
        ts = self.thetas[item*self.batch_size: (item+1)*self.batch_size]
        if self.fix_beta_zero:
            bs = np.zeros((self.batch_size, 10), dtype=np.float64)
        else:
            bs = self.betas[item*self.batch_size: (item+1)*self.batch_size]
        return {'joints': js, 'thetas': ts, 'betas': bs}
        
    def __len__(self):
        return self.length

"""
    https://en.wikipedia.org/wiki/Rotation_R
    Inverse Rodrigues' rotation formula that turns rotation
    matrix into axis-angle tensor in a batch-ed manner.

    Parameter:
    ----------
    R: Rotation matrix of shape [batch_size * angle_num, 3, 3].
    
    Return:
    -------
    Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
"""
def inv_rodrigues(R):
    axis = torch.stack((
        R[:,2,1] - R[:,1,2],
        R[:,0,2] - R[:,2,0],
        R[:,1,0] - R[:,0,1],
        ), dim=1
    )
    # Angle, beware if R is close to I
    # (theta close to 2*K*pi, imprecise arccos)
    eps = 1e-6
    axis_norm = torch.norm(axis, dim=1)
    eps_norm = eps * torch.zeros_like(axis_norm)
    axis_norm = torch.where(axis_norm > eps, axis_norm, eps_norm)
    
    trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
    angle = torch.atan2(axis_norm, trace-1)
    
    # Angle is not unique, consider fix it into [0, 2pi]
    
    # Normalise the axis.
    axis /= axis_norm.unsqueeze(dim=1)
    
    # Return the data in compressed format [ax,ay,az]
    return axis * angle.unsqueeze(dim=1)
    
def unit_test_inv_rodrigues(model=None):
    if model is None:
        model = SMPLModel(device=device, model_path = 'model_24_joints.pkl')
    
    print('left inverse')
    theta = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 3)\
          .type(torch.float64).to(device).view(-1,1,3)
    
    theta_recon = inv_rodrigues(model.rodrigues(theta))
    print(theta_recon.shape)
    
    # Input theta must be pre-processed so that the theta value lies in (-pi, pi]
    
    print(torch.max(torch.norm(theta.squeeze() - theta_recon, dim=1)))
    

if __name__ == '__main__':
    pose_size = 72
    beta_size = 10
    batch_size = 64
    
    np.random.seed(9608)
    device = torch.device('cuda')
    model = SMPLModel(device=device, model_path = 'model_24_joints.pkl')
    #print(model.kintree_table)
    #print(model.parent)
    
    dataset = Joint2SMPLDataset('train_dataset_24_joints_1.0.pickle', batch_size, fix_beta_zero=True)
    
    #unit_test_inv_rodrigues(model=model)
    
    '''
        Regression rotation R
    '''
    sample_num = 2000
    item = np.random.choice(len(dataset), sample_num, replace=False)
    js = torch.from_numpy(dataset.joints[item]).to(device)
    ts = torch.from_numpy(dataset.thetas[item]).to(device)
    bs = torch.zeros((sample_num, 10), dtype=torch.float64, device=device)
    # relative rotation R, from root to parents
    # Use a directed graph to record kinematic tree.
    # After recovering rotation R, convert it to axis angles.
    
    # Start from the groin (root joint #0)
    # axis-angle representation. 
    
    standard_joint = torch.from_numpy(
        np.loadtxt('standard_joint.xyz')
    ).to(device)
    
    #print(standard_joint)
    
    theta_3 = ts[:, :3]    # sample_num * 3
    joint_0 = js[:, 0, :]    # sample_num * 3, joint_0 coords
    
    #print(joint_0[0])
    #print(theta_3[0])
    
    # Basic routine:
    #   Denote the joint_0 of the standard model and the deformed model
    #   to be x and y, then (x \cross y) gives the rotation angle and 
    #   the angle is easy to calculate this way.
    
    # Try on sample #0
    # Forward
    print('Forward verificaition:')
    jx = standard_joint[0]
    jy = joint_0[0]
    theta = theta_3[0].view(1,1,-1)
    print(theta)
    R = model.rodrigues(theta).squeeze()
    print('Rx:', R @ jx)
    print('y:', jy)
    
    # Backward
    print('Backward Estimation:')
    jx = standard_joint[0]
    jy = joint_0[0]
    print(jx, jy)
    axis = torch.cross(jx, jy)
    axis /= torch.norm(axis)
    dot = torch.dot(jx,jy)
    dot /= (torch.norm(jx) * torch.norm(jy))
    angle = torch.acos(dot)
    axis *= angle
    print(' regression:', axis)
    print(' GT:', theta_3[0])
    #print(jx.shape, jy.shape)
    
    
    