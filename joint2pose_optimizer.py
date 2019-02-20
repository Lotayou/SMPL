import torch
from torch import nn
from torch.optim import SGD, Adam
import pickle
import numpy as np
from smpl_torch_batch import SMPLModel
import os
from time import time

'''
    joint2pose_optimizer: Testing the backward capability of SMPLModel
    @ Given a randomly generated human mesh, try to find the thetas by matching joints only.
'''



if __name__ == '__main__':
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #print(device)

    pose_size = 72
    beta_size = 10

    np.random.seed(9608)
    model = SMPLModel(device=device)
    
    if not os.path.isdir('joint2pose_result'):
        os.makedirs('joint2pose_result')
    
    loss_op = nn.L1Loss()
    betas = torch.zeros((1, beta_size), dtype=torch.float64, device=device)
    trans = torch.zeros((1, 3), dtype=torch.float64, device=device)
    
    for i in range(10):
        print('Test case %d:' % (i+1))
        real_pose = torch.from_numpy((np.random.rand(1, pose_size) - 0.5) * 0.5)\
              .type(torch.float64).to(device)
        real_result, real_joints = model(betas, real_pose, trans)
        model.write_obj(real_result[0].detach().cpu().numpy(), 'joint2pose_result/real_mesh_{}.obj'.format(i))
        
        # Initialize a pose from zero and optimize it
        
        test_pose = torch.zeros((1, pose_size), dtype=torch.float64, device=device, requires_grad=True)
        #optimizer = SGD(iter([test_pose]), lr=0.001, momentum=0.2)
        optimizer = Adam(iter([test_pose]), lr=0.0001, betas=(0.5,0.999))
        
        s = time()
        prev_loss = None
        for step in range(1000):
            _, test_joints = model(betas, test_pose, trans)
            loss = loss_op(test_joints, real_joints)
            print('Step {:03d}: loss: {:10.6f}'.format(step, loss.data.item()))
            cur_loss = loss.data.item()
            if prev_loss is not None and cur_loss > prev_loss:
                break
            loss.backward()
            optimizer.step()
            prev_loss = cur_loss
            
        print('Time: ', time() - s)
        
        test_result, test_joints = model(betas, test_pose, trans)
        model.write_obj(test_result[0].detach().cpu().numpy(), 'joint2pose_result/test_mesh_{}.obj'.format(i))
        print('Real joints:\n', real_joints)
        print('Test joints:\n', test_joints)
        
        print('Real pose:\n', real_pose)
        print('Test pose:\n', test_pose)
        
        np.savetxt('joint2pose_result/real_joints_{}.xyz'.format(i), real_joints[0].detach().cpu().numpy().reshape(19,3), delimiter=' ')
        np.savetxt('joint2pose_result/test_joints_{}.xyz'.format(i), test_joints[0].detach().cpu().numpy().reshape(19,3), delimiter=' ')
