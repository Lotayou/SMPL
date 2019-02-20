import torch
from torch import nn
import os
from torch.optim import Adam
import numpy as np
import pickle
from train_regressor_joints_recon_loss import Regressor, Joint2SMPLDataset
from smpl_torch_batch import SMPLModel
from renderer import draw_skeleton
from cv2 import imwrite
from torch.utils.data import Dataset, DataLoader

class Pose2MeshModel(nn.Module):
    def __init__(self):
        super(Pose2MeshModel, self).__init__()
        if torch.cuda.is_available():
            self.reg = Regressor(hidden_dim=512).cuda()
            self.smpl = SMPLModel(device=torch.device('cuda'))
        else:
            self.reg = Regressor(hidden_dim=512).cpu()
            self.smpl = SMPLModel(device=torch.device('cpu'))
           
        ckpt_path = './checkpoints_recon_loss_hl_3_hd_512'
        state_dict = torch.load('%s/regressor_100.pth' % (ckpt_path))
        self.reg.load_state_dict(state_dict)
            
    def forward(self, input):
        trans = torch.zeros((input.shape[0], 3), device=input.device)
        thetas, betas = self.reg(input)
        mesh, joints = self.smpl(betas, thetas, trans)
        return mesh, joints
        
    def evaluate(self, input, save_dir):
        mesh, joints = self.forward(input)
        self.smpl.write_obj(mesh[0].detach().cpu().numpy(), save_dir)
        np.savetxt('recon_pose.xyz', joints[0].detach().cpu().numpy().reshape(19,3), delimiter=' ')
       
       
if __name__ == '__main__':
    torch.backends.cudnn.enabled=True
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda')
    model = Pose2MeshModel()
    
    dataset = Joint2SMPLDataset('train_dataset.pickle', batch_size=64)
    index = np.random.randint(0, dataset.length)
    joints_npy = dataset[index:index+2]['joints']
    
    # render joints
    joints = torch.as_tensor(joints_npy, device=device)
    # image = np.zeros((512, 512, 3), dtype=np.uint8)
    # joints_image = draw_skeleton(image, joints_npy[0].reshape(3, 19))
    # imwrite('Input_skeleton.png', joints_image)
    np.savetxt('test_pose.xyz', joints_npy[0].reshape(19,3), delimiter=' ')
    model.evaluate(joints, 'test_mesh.obj')
        