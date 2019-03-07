import torch
from torch import nn
import os
from torch.optim import Adam
import numpy as np
import pickle
from train_acos_regressor_24_joints import AcosRegressor, Joint2SMPLDataset
from smpl_torch_batch import SMPLModel
from cv2 import imwrite
from torch.utils.data import Dataset, DataLoader

class Pose2MeshModel(nn.Module):
    def __init__(self):
        super(Pose2MeshModel, self).__init__()
        if torch.cuda.is_available():
            self.reg = AcosRegressor(hidden_dim=256).cuda()
            self.smpl = SMPLModel(device=torch.device('cuda'),
                model_path = './model_24_joints.pkl',
                    simplify=True
            )
        else:
            self.reg = AcosRegressorRegressor(hidden_dim=256).cpu()
            self.smpl = SMPLModel(device=torch.device('cpu'),
                model_path = './model_24_joints.pkl',
                    simplify=True
            )
           
        ckpt_path = './checkpoints_0303_24_joints'
        state_dict = torch.load('%s/regressor_040.pth' % (ckpt_path))
        self.reg.load_state_dict(state_dict)
            
    def forward(self, input):
        trans = torch.zeros((input.shape[0], 3), device=input.device)
        betas = torch.zeros((input.shape[0], 10), device=input.device)
        thetas = self.reg(input)
        print('Estimated theta:\n', thetas.detach().cpu().numpy())
        mesh, joints = self.smpl(betas, thetas, trans)
        return mesh, joints
        
    def evaluate(self, input, save_dir):
        mesh, joints = self.forward(input)
        self.smpl.write_obj(mesh[0].detach().cpu().numpy(), save_dir)
        np.savetxt('recon_pose.xyz', joints[0].detach().cpu().numpy().reshape(24,3), delimiter=' ')
       
       
if __name__ == '__main__':
    torch.backends.cudnn.enabled=True
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda')
    model = Pose2MeshModel()
    
    dataset = Joint2SMPLDataset('train_dataset_24_joints_1.0.pickle', batch_size=64, fix_beta_zero=True)
    index = np.random.randint(0, dataset.length)
    joints_npy = dataset[index:index+2]['joints']
    thetas_npy = dataset[index:index+2]['thetas']
    
    # render joints
    joints = torch.as_tensor(joints_npy, device=device)
    print('Ground truth theta:\n', thetas_npy)
    # image = np.zeros((512, 512, 3), dtype=np.uint8)
    # joints_image = draw_skeleton(image, joints_npy[0].reshape(3, 19))
    # imwrite('Input_skeleton.png', joints_image)
    np.savetxt('input_pose.xyz', joints_npy[0].reshape(24,3), delimiter=' ')
    model.evaluate(joints, 'recon_mesh.obj')

