'''
    linear_regression.py: Try to approximate empirical solution of pose-to-theta
        with correspondence observed from interpolation results.
    
    Empirically, we guess any theta should have the form:
    
        $$ theta_i = acos(<p_j - p_k, p_l - p_k>) * w_i + b_i $$
        
    where w_i and b_i will be estimated using least square regression.
'''

from smpl_torch_batch import SMPLModel
import numpy as np
import torch
import os
        
'''
    SMPLModelv2: An extension to the original SMPLModel with joint2theta inference
'''     
class SMPLModelv2(SMPLModel):
    def __init__(self, device=None, model_path='./model.pkl', simplify=True):
        super(SMPLModelv2, self).__init__(device, model_path, simplify)
        self.J0 = torch.mm(self.J_regressor, self.v_template)
        
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
        print('Axis:\n', axis)
        # Angle, beware if R is close to I
        # (theta close to 2*K*pi, imprecise arccos)
        eps = 1e-6
        axis_norm = torch.norm(axis, dim=1)
        eps_norm = eps * torch.ones_like(axis_norm)
        axis_norm = torch.where(axis_norm > eps_norm, axis_norm, eps_norm)
        print(axis_norm)
        
        trace = R[:,0,0] + R[:,1,1] + R[:,2,2]
        angle = torch.atan2(axis_norm, trace-1)
        print(angle)
        
        # Angle is not unique, consider fix it into [0, 2pi]
        
        # Normalise the axis.
        axis /= axis_norm.unsqueeze(dim=1)
        
        # Return the data in compressed format [ax,ay,az]
        return axis * angle.unsqueeze(dim=1)
        
    # 20190305: unit test passed!
    # 20190310: unit test for theta==0
    def unit_test_inv_rodrigues(self):
        print('Unit test inv rodrigues:')
        print('   Zero theta:')
        theta = torch.zeros((2, 1, 3), dtype=torch.float64, device=self.device)
        theta_recon = self.inv_rodrigues(self.rodrigues(theta))
        print('Reconstruction error: ', 
            torch.max(torch.norm(theta.squeeze() - theta_recon, dim=1)))
            
        
        # print('   Random theta:')
        # theta = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 3)\
              # .type(torch.float64).to(device).view(-1,1,3)
        
        # theta_recon = self.inv_rodrigues(self.rodrigues(theta))
        
        # Input theta must be pre-processed so that the theta value lies in (-pi, pi]
        # print('Reconstruction error: ', 
            # torch.max(torch.norm(theta.squeeze() - theta_recon, dim=1)))
            
        
            
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
        
    '''
        R2theta: get thetas from regressed global rotations
    '''
    def R2theta(self, gR):
        batch_size = gR.shape[0]
        
        # backward transversal from kinematic trees.
        Rs = [gR[:, 0]]
        for i in range(1, self.kintree_table.shape[1]):
            # Solve the relative rotation matrix at current joint
            # Apply inverse rotation for all subnodes of the tree rooted at current joint
            # Update: Compute quick inverse for rotation matrices (actually the transpose)
            Rs.append(torch.bmm(gR[:, self.parent[i]].transpose(1,2), gR[:, i]))
            
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
        solveR(u,v): find a rotation matrix that takes point u to v
        both u and v are [N, 3] tensors with ||u[i]|| == ||v[i]|| == 1
    '''
    def solveR(self, u, v):
    
        def ssc(v):
            #print(v)
            Os = torch.zeros(v.shape[0], dtype=torch.float64).to(v.device)
            m = torch.stack((
                Os, -v[:,2], v[:,1],
                v[:,2], Os, -v[:,0],
                -v[:,1], v[:,0], Os), dim=1
            )
            V = torch.reshape(m, (-1, 3, 3))
            #print(V)
            return V
    
        eps = 1e-8
        n = torch.cross(u, v)
        sin_uv = torch.norm(n, dim=1)
        cos_uv = torch.sum(u*v, dim=1)
        # print(cos_uv)
        # Avoid the case when cos_uv = -1
        cos_uv = torch.max(cos_uv, torch.tensor(
            -1+eps, dtype=torch.float64, device=self.device
        ))
        I = torch.eye(3, dtype=torch.float64, device=self.device).expand(u.shape[0], -1, -1)
        N = ssc(n)
        N2 = torch.bmm(N, N)
        wN2 = (1 / (1 + cos_uv)).expand(3,3,-1).transpose(0,2)
        R = I + N + wN2 * N2
        return R

    def unit_test_solveR(self):
        print('Unit test solveR:')
        u = torch.rand((32,3), dtype=torch.float64, device=self.device)
        v = torch.rand((32,3), dtype=torch.float64, device=self.device)
        nu = torch.norm(u, dim=1, keepdim=True)
        u /= nu
        nv = torch.norm(v, dim=1, keepdim=True)
        v /= nv
        
        R = self.solveR(u, v)
        RRt = torch.bmm(R, R.transpose(1,2))
        for i in range(R.shape[0]):
           print(torch.mm(R[i], R[i].transpose(0,1)))
        print("Orthogonal check: R * R' =\n", RRt)
        
        v_ = torch.bmm(R, u.view(-1,3,1)).squeeze(dim=2)
        print('|Ru-v|:', torch.norm(v_ - v, dim=1))
        
        
    '''
        regressG: directly regress the most suitable G' to translate 
        original skeleton J0 to given input J in a batched manner
        (i.e. input [N * 24 * 3]
    '''
    def regressR(self, j):
        # Regress 24 global rigid transformation matrices that maps skeleton J0 to J
        # calculate global translation vector
        batch_size = j.shape[0]
        j0 = self.J0.expand_as(j)
        
        # Normalize j and j0 to make bones unit length.
        parent = torch.tensor([self.parent[i] for i in range(1,24)], device=self.device)
        dj0 = j0[:, 1:24] - j0[:, parent]
        ndj0 = torch.norm(dj0, dim=2, keepdim=True)
        dj0 /= ndj0
        
        dj = j[:, 1:24] - j[:, parent]
        ndj = torch.norm(dj, dim=2, keepdim=True)
        dj /= ndj
        
        # What about hands and feet? Ignore it?
        # Adding additional control points.
        batch_I3 = torch.eye(3, dtype=torch.float64, device=self.device).expand(batch_size, -1, -1)
        Rs = [None] * 24
        for i in range(1,24):
            p_i = self.parent[i]
            if Rs[p_i] is None:
                Rs[p_i] = self.solveR(dj0[:, i-1], dj[:, i-1])
        
        # Set unresolved rotations to I_3
        for i in range(24):
            if Rs[i] is None:
                Rs[i] = batch_I3
            
        Rs = torch.stack(Rs, dim=1)
        return Rs

    def unit_test_regressR(self):
        print('Unit test regressR')
        # Only regress G0, the rest can be solved numerically.
        real_thetas = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 1.5)\
          .type(torch.float64).to(self.device)
        
        '''
            Fix global rotation, change local rotation
        '''
        # Test if change R[23] affects joint 23, and the shape of finger tips?
        # Check passed, change leaf rotation does not affect leaf joints, but do affect the body shapes
        # Consider adding another control point at head to determine head rotation.
        undefined = [30, 31, 32, 33, 34, 35, 45, 46, 47, 66, 67, 68, 69, 70, 71]
        index = torch.zeros(72, dtype=torch.int, device=self.device)
        index[undefined] = 1
        for j in range(1,32):
            real_thetas[j] = torch.where(index == 0, real_thetas[0], real_thetas[j])
        
        #print('thetas:', real_thetas)
        
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
        globalR = self.regressR(joints)
        return self.R2theta(globalR)
        
        
    def unit_test_joint2theta(self):
        print('Unit test joint2theta')
        real_thetas = torch.from_numpy((np.random.rand(32, pose_size) - 0.5) * 1)\
          .type(torch.float64).to(self.device)
        betas = torch.from_numpy(np.zeros((32, beta_size))) \
          .type(torch.float64).to(self.device)
        trans = torch.from_numpy(np.zeros((32, 3))).type(torch.float64).to(self.device)
        real_meshes, real_joints = self.forward(betas, real_thetas, trans)
        # check pass, bone length roughly hold (+- 2% error)
        #print('norm J:\n', torch.norm(joints[:,1] - joints[:,0], dim=1))
        
        fake_Rs, fake_thetas = self.joint2theta(real_joints)
        print('theta Reconstruction error: ', 
           torch.norm(real_thetas - fake_thetas, dim=1))
        print('theta Batch #0 residual: ',
            real_thetas[0] - fake_thetas[0])
            
        fake_meshes, fake_joints = self.forward(betas, fake_thetas, trans)
        print('Joint residual:', 
            torch.max(torch.norm(real_joints - fake_joints, dim=(2)), dim=1))
            
        for i in range(32):
            model.write_obj(real_meshes[i].detach().cpu().numpy(), './joint2theta_test/real_{}.obj'.format(i))
            model.write_obj(fake_meshes[i].detach().cpu().numpy(), './joint2theta_test/fake_{}.obj'.format(i))
            np.savetxt('./joint2theta_test/real_{}.xyz'.format(i), real_joints[i].detach().cpu().numpy(), delimiter=' ')
            np.savetxt('./joint2theta_test/fake_{}.xyz'.format(i), fake_joints[i].detach().cpu().numpy(), delimiter=' ')
        
if __name__ == '__main__':
    pose_size = 72
    beta_size = 10
    
    np.random.seed()
    device = torch.device('cuda')
    model = SMPLModelv2(device=device, model_path = 'model_24_joints.pkl',
                    simplify=True)
    
    model.unit_test_inv_rodrigues()
    # model.unit_test_G2theta()
    # model.unit_test_solveR()
    # model.unit_test_joint2theta()
    