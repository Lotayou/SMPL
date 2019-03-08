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

        
'''
    SMPLModelv2: An extension to the original SMPLModel with joint2theta inference
'''     
class SMPLModelv2(SMPLModel):
    def __init__(self, device=None, model_path='./model.pkl', simplify=False):
        super(SMPLModelv2, self).__init__(device, model_path, simplify)
        self.J0 = torch.mm(self.J_regressor, self.v_template)
        #print('J0:\n', self.J0)
        #print('norm J0:\n', torch.norm(self.J0[1] - self.J0[0]))
        
    """
        https://en.wikipedia.org/wiki/Rotation_R
        Inverse Rodrigues' rotation formula that turns rotation
        matrix into axis-angle tensor in a batch-ed manner.

        Parameter:
        ----------
        R: Rotation matrix of shape [batch_size * joint_num, 3, 3].
        
        Return:
        -------
        Axis-angle rotation tensor of shape [batch_size * joint_num, 1, 3].
    """
    @staticmethod
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
        
    # 20190305: unit test passed!
    def unit_test_inv_rodrigues(self):
        print('Unit test inv rodrigues:')
        theta = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 3)\
              .type(torch.float64).to(device).view(-1,1,3)
        
        theta_recon = self.inv_rodrigues(self.rodrigues(theta))
        print(theta_recon.shape)
        
        # Input theta must be pre-processed so that the theta value lies in (-pi, pi]
        print('Reconstruction error: ', 
            torch.max(torch.norm(theta.squeeze() - theta_recon, dim=1)))
            
    '''
        G2theta: calculate theta from input G terms
            should be the inverse of SMPLModel::theta2G
        
        Parameter:
        ----------
        G: A tensor of shape [batch_size, joint_num, 4, 4]
        

        Return:
        ------
        Rs: Relative rotation matrices at joints, shape[batch_size, joint_num, 3, 3]
        thetas: A tensor of shape [batch_size, joint_num * 3]
    '''
    def G2theta(self, G):
        batch_size = G.shape[0]
        # Retrieve G from G'
        R = G[:, :, 0:3, 0:3]
        j = G[:, :, 0:3, 3:4]
        I = torch.eye(3, dtype=torch.float64, device=self.device).expand_as(R)
        G[:, :, 0:3, 3:4] = torch.matmul(torch.inverse(I-R), j)
        
        # backward transversal from kinematic trees.
        Rs = [R[:, 0]]
        for i in range(1, self.kintree_table.shape[1]):
            # Solve the relative rotation matrix at current joint
            # Apply inverse rotation for all subnodes of the tree rooted at current joint
            # Update: Compute quick inverse for rotation matrices (actually the transpose)
            Rs.append(torch.bmm(R[:, self.parent[i]].transpose(1,2), R[:, i]))
            
        Rs = torch.stack(Rs, dim=1)
        thetas = self.inv_rodrigues(Rs.view(-1,3,3)).reshape(batch_size, -1)
        return Rs, thetas
    
    # 20190307: unit test passed!
    def unit_test_G2theta(self):    
        print('Unit test G2theta:')
        J = self.J0.expand(32,-1,-1)
        # generate some theta
        real_thetas = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 2)\
          .type(torch.float64).to(self.device)
        G, R_cube_big = self.theta2G(real_thetas, J)
        #print('G:\n',G[0])
        R, recon_thetas = self.G2theta(G)
        
        print('R reconstruction error: ', 
            torch.max(torch.norm(R_cube_big - R, dim=(2,3))))
        print('theta reconstruction error: ', 
            torch.max(torch.norm(real_thetas - recon_thetas, dim=1)))

    '''
        findR(u,v): find a rotation matrix that takes point u to v
        both u and v are [N, 3] tensors with ||u[i]|| == ||v[i]|| != 0
    '''
    def findR(u, v):
        u_cross_v = torch.cross(u, v)
        norm_n = torch.norm(u_cross_v, 0)
        # TODO: finish up.
        
        
    '''
        regressG: directly regress the most suitable G' to translate 
        original skeleton J0 to given input J in a batched manner
        (i.e. input [N * 24 * 3]
    '''
    def regressG(self, j):
        # Regress 24 global rigid transformation matrices that maps skeleton J0 to J
        # calculate global translation vector
        j0 = self.J0.expand_as(j)
        
        # cache tensor
        
        # Solve Global rotation G
        # 20190308: TODO: normalize j as j0 to make bones equal length.
        
        
        pass

    def unit_test_regressG(self):
        print('Unit test regressG')
        # Only regress G0, the rest can be solved numerically.
        real_thetas = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 1)\
          .type(torch.float64).to(self.device)
        '''
            Fix global rotation, change local rotation
        '''
        # local_theta = real_thetas[0,:3]
        # for j in range(1,32):
            # real_thetas[j, :3] = local_theta
        
        # print('thetas:', real_thetas)
        
        betas = torch.from_numpy(np.zeros((32, beta_size))) \
          .type(torch.float64).to(self.device)
        trans = torch.from_numpy(np.zeros((32, 3))).type(torch.float64).to(self.device)
        
        meshes, joints = self.forward(betas, real_thetas, trans)
        #reg_G = self.regressG(joints)

        v_shaped = torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template
        J = torch.matmul(self.J_regressor, v_shaped)
        G, R_cube_big = self.theta2G(real_thetas, J)  # pre-calculate G terms for skinning 
        # print(G[0,0])
        # R00 = G[0,0,:3,:3]
        # print('j0:',self.J0[0])
        # j0_ = torch.matmul(
            # torch.eye(3, dtype=torch.float64, device =self.device) - R00,
            # self.J0[0]
        # )
        # print('(I-R_0)J0:',j0_)
        
        # What if we directly apply G on J?
        J_1 = torch.cat(
            (J, torch.ones((32, J.shape[1], 1), dtype=torch.float64).to(self.device)), dim=2
        ).reshape(32,-1,4,1)
        fake_joints = torch.matmul(G, J_1)
        fake_joints = torch.reshape(fake_joints, (32, -1, 4))[:,:,:3]
        #print('G_0J0:',fake_joints[0,0])
        #print('real joint_0:',joints[0,0])
        
        # Test if directly regress joints from G works...
        # 20190308: Good approximation, visually undiscernable.
        for i in range(32):
            model.write_obj(meshes[i].detach().cpu().numpy(), './joint_test_0308/real_{}.obj'.format(i))
            np.savetxt('./joint_test_0308/real_{}.xyz'.format(i), joints[i].detach().cpu().numpy(), delimiter=' ')
            np.savetxt('./joint_test_0308/fake_{}.xyz'.format(i), fake_joints[i].detach().cpu().numpy(), delimiter=' ')
            
        
        
        
    '''
        joint2theta: Regress theta parameters from given joints
        Note: This method assumes that beta are fixed to 0.
    '''
    def joint2theta(self, joints):
        # regression: joints to G
        G = self.regressG(joints)
        _, thetas = self.G2theta(G)
        return thetas
        
        
    def unit_test_joint2theta(self):
        print('Unit test joint2theta')
        real_thetas = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 2)\
          .type(torch.float64).to(self.device)
        betas = torch.from_numpy(np.zeros((32, beta_size))) \
          .type(torch.float64).to(self.device)
        trans = torch.from_numpy(np.zeros((32, 3))).type(torch.float64).to(self.device)
        _, joints = self.forward(betas, real_thetas, trans)
        # check pass, bone length roughly hold (+- 2% error)
        #print('norm J:\n', torch.norm(joints[:,1] - joints[:,0], dim=1))
        
        # infer_thetas = self.joint2theta(joints)
        # print('Reconstruction error: ', 
           # torch.max(torch.norm(real_thetas - infer_thetas, dim=1)))
    
    
if __name__ == '__main__':
    pose_size = 72
    beta_size = 10
    batch_size = 64
    
    #np.random.seed(9608)
    np.random.seed()
    device = torch.device('cuda')
    model = SMPLModelv2(device=device, model_path = 'model_24_joints.pkl',
                    simplify=True)
    #print('k table: ', model.kintree_table)
    #print('parent: ', model.parent)
    
    model.unit_test_inv_rodrigues()
    model.unit_test_G2theta()
    model.unit_test_regressG()
    #model.unit_test_joint2theta()
    
    # dataset = Joint2SMPLDataset('train_dataset_24_joints_1.0.pickle', batch_size, fix_beta_zero=True)
    # sample_num = 2000
    # item = np.random.choice(len(dataset), sample_num, replace=False)
    # js = torch.from_numpy(dataset.joints[item]).to(device)
    # ts = torch.from_numpy(dataset.thetas[item]).to(device)
    # bs = torch.zeros((sample_num, 10), dtype=torch.float64, device=device)
    
    
    
    
    