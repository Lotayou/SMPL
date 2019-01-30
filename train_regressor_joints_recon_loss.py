import torch
from torch import nn
import os
from torch.optim import Adam
import numpy as np
import pickle
from smpl_torch_batch import SMPLModel
from torch.utils.data import Dataset, DataLoader


class Joint2SMPLDataset(Dataset):
    '''
        Regression Data with Joint and Theta, Beta.
        Predict Pose angles and Betas from input joints.
        Train/val: 1:1
    '''
    def __init__(self, pickle_file, batch_size=64):
        super(Joint2SMPLDataset, self).__init__()
        assert(os.path.isfile(pickle_file))
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        
        self.thetas = dataset['thetas']
        self.betas = dataset['betas']
        self.joints = dataset['joints'].reshape(-1, 19*3)
        
        print(self.joints.shape)
        self.batch_size = batch_size
        self.length = self.joints.shape[0] // 2
        print(self.length)
        
    def __getitem__(self, item):
        js = self.joints[item]
        ts = self.thetas[item]
        bs = self.betas[item]
        return {'joints': js, 'thetas': ts, 'betas': bs}
        
    def rand_val_batch(self):
        length = self.length // self.batch_size
        item = np.random.randint(0, length) + length
        js = self.joints[item*self.batch_size: (item+1)*self.batch_size]
        ts = self.thetas[item*self.batch_size: (item+1)*self.batch_size]
        bs = self.betas[item*self.batch_size: (item+1)*self.batch_size]
        return {'joints': js, 'thetas': ts, 'betas': bs}
        
    def __len__(self):
        return self.length


class Regressor(nn.Module):
    def __init__(self, hidden_dim=256, indim=57, thetadim=72, betadim=10,
                batch_size=64, hidden_layer=3, use_dropout=False):
        super(Regressor, self).__init__()
        model = [
            nn.Linear(indim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        ]
        for i in range(hidden_layer):
            model.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
            ])
            if use_dropout:
                model.append(nn.Dropout(0.5))
                
        self.feature_extractor = nn.Sequential(*model)
        self.theta_predictor = nn.Linear(hidden_dim, thetadim)
        self.beta_predictor = nn.Linear(hidden_dim, betadim)
        
    def forward(self, x):
        h = self.feature_extractor(x)
        theta = self.theta_predictor(h)
        beta = self.beta_predictor(h)
        return theta, beta
        

if __name__ == '__main__':
    torch.backends.cudnn.enabled=True
    batch_size = 64
    max_batch_num = 100
    dataset = Joint2SMPLDataset('train_dataset.pickle', batch_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda')
    reg = Regressor(batch_size=batch_size).cuda()
    smpl = SMPLModel(device=device)
    loss_op = nn.L1Loss()
    optimizer = Adam(reg.parameters(), lr=0.001, betas=(0.5,0.999), weight_decay=1e-5)
    
    batch_num = 0
    ckpt_path = './checkpoints_recon_loss_hl_3'
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    if batch_num > 0 and os.path.isfile('%s/regressor_%03d.pth' % (ckpt_path, batch_num)):
        state_dict = torch.load_state_dict('%s/regressor_%03d.pth' % (ckpt_path, batch_num))
        reg.load(state_dict)
        
    file = open('train_log_recon_loss_hl_3.txt', 'w')

    trans = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)

    while batch_num <= max_batch_num:
        batch_num += 1
        print('Epoch %03d: training...' % batch_num)
        reg.train()
        for (i, data) in enumerate(dataloader):
            joints = torch.as_tensor(data['joints'], device=device)
            thetas = torch.as_tensor(data['thetas'], device=device)
            betas = torch.as_tensor(data['betas'], device=device)
            
            pred_thetas, pred_betas = reg(joints)
            _, recon_joints = smpl(pred_betas, pred_thetas, trans)
            loss_joints = loss_op(recon_joints.contiguous().view(batch_size, -1), joints)
            optimizer.zero_grad()
            loss_joints.backward()
            optimizer.step()
            
            if i % 32 == 0:
                print('batch %04d: loss: %10.6f' % (i, loss_joints.data.item()))
            
        print('Validation: ')
        reg.eval()
        data = dataset.rand_val_batch()
        joints = torch.as_tensor(data['joints'], device=device)
        thetas = torch.as_tensor(data['thetas'], device=device)
        betas = torch.as_tensor(data['betas'], device=device)
        with torch.no_grad():
            pred_thetas, pred_betas = reg(joints)
            _, recon_joints = smpl(pred_betas, pred_thetas, trans)
            loss_joints = loss_op(recon_joints.contiguous().view(batch_size, -1), joints)
            line = 'Validation: loss_theta: %10.6f' % loss_joints.data.item()
            print(line)
            file.write(line+'\n')
            
        if batch_num % 5 == 0:
            print('Save models...')
            torch.save(reg.state_dict(), '%s/regressor_%03d.pth' % (ckpt_path, batch_num))
    file.close()
