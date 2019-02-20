import torch
from torch import nn
import os
from torch import optim
import numpy as np
import pickle
from smpl_torch_batch import SMPLModel
from torch.utils.data import Dataset, DataLoader
from sys import platform

class Joint2SMPLDataset(Dataset):
    '''
        Regression Data with Joint and Theta, Beta.
        Predict Pose angles and Betas from input joints.
        Train/val: 1:1
    '''
    def __init__(self, pickle_file, batch_size=64,fix_beta_zero=False):
        super(Joint2SMPLDataset, self).__init__()
        assert(os.path.isfile(pickle_file))
        with open(pickle_file, 'rb') as f:
            dataset = pickle.load(f)
        
        self.thetas = dataset['thetas']
        self.joints = dataset['joints'].reshape(-1, 19*3)
        self.fix_beta_zero = fix_beta_zero
        if not fix_beta_zero:
            self.betas = dataset['betas']
        
        print(self.joints.shape)
        self.batch_size = batch_size
        self.length = self.joints.shape[0] // 2
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
        item = np.random.randint(0, length) + length
        js = self.joints[item*self.batch_size: (item+1)*self.batch_size]
        ts = self.thetas[item*self.batch_size: (item+1)*self.batch_size]
        if self.fix_beta_zero:
            bs = np.zeros((self.batch_size, 10), dtype=np.float64)
        else:
            bs = self.betas[item*self.batch_size: (item+1)*self.batch_size]
        return {'joints': js, 'thetas': ts, 'betas': bs}
        
    def __len__(self):
        return self.length

        
class ResBlock1d(nn.Module):
    def __init__(self, indim=256, outdim=None, use_dropout=False):
        super(ResBlock1d, self).__init__()
        if outdim is None:
            outdim = indim
            
        self.indim = indim
        self.outdim = outdim
        model = [
            nn.Linear(indim, indim),
            nn.BatchNorm1d(indim),
            nn.LeakyReLU(0.2)
        ]
        if use_dropout:
            model.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*model)
        
        if outdim != indim:
            self.linear = nn.Linear(indim, outdim)

    def forward(self, x):
        out = x + self.model(x)
        if self.outdim != self.indim:
            out = self.linear(out)
        return out


class ResidualRegressor(nn.Module):
    def __init__(self, hidden_dim=256, indim=57, thetadim=72, betadim=10,
                batch_size=64, hidden_layer=3, use_dropout=False):
        super(ResidualRegressor, self).__init__()
        model = [
            nn.Linear(indim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2)
        ]
        for i in range(hidden_layer):
            model += [ResBlock1d(indim=hidden_dim, use_dropout=use_dropout)]

        self.feature_extractor = nn.Sequential(*model)
        self.theta_predictor = nn.Linear(hidden_dim, thetadim)
        #self.beta_predictor = nn.Linear(hidden_dim, betadim)
        
    def forward(self, x):
        h = self.feature_extractor(x)
        theta = self.theta_predictor(h)
        #beta = self.beta_predictor(h)
        return theta
        

class AcosRegressor(nn.Module):
    def __init__(self, hidden_dim=256, indim=57, thetadim=72, betadim=10,
                batch_size=64, hidden_layer=3, use_dropout=False):
        super(AcosRegressor, self).__init__()
        self.limbs_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18],
            [1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, 14, 13, 13, 15, 16]
        ], dtype=torch.long)
        
        model = [ResBlock1d(indim=324+57, outdim=hidden_dim)]
        for i in range(hidden_layer):
            model += [ResBlock1d(indim=hidden_dim, use_dropout=use_dropout)]
        model += [nn.Linear(hidden_dim, thetadim)]
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        # expect N * 19 * 3
        vec = x[:, self.limbs_index[0], :] - x[:, self.limbs_index[0], :]
        prod = torch.bmm(vec, vec.transpose(1, 2))
        angles = torch.acos(prod).view(-1, 324)
        features = torch.cat((x.view(-1, 57), angles), dim=1)
        return self.model(features)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    torch.backends.cudnn.enabled=True
    batch_size = 128
    max_batch_num = 100
    #dataset = Joint2SMPLDataset('train_dataset.pickle', batch_size)
    dataset = Joint2SMPLDataset('train_dataset_fix_beta_zero.pickle', batch_size, fix_beta_zero=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda')
    reg = AcosRegressor(batch_size=batch_size).cuda()
    smpl = SMPLModel(device=device)
    loss_op = nn.L1Loss()
    optimizer = optim.Adam(reg.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.333, patience=1, verbose=True)
    
    batch_num = 0
    ckpt_path = 'checkpoints_0219_decay_1_3_100_epochs'
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    if batch_num > 0 and os.path.isfile('%s/regressor_%03d.pth' % (ckpt_path, batch_num)):
        state_dict = torch.load_state_dict('%s/regressor_%03d.pth' % (ckpt_path, batch_num))
        reg.load(state_dict)

    # copy current file into checkpoint folder to record parameters, ugly.
    if platform == 'linux':
        cmd = 'cp train_acos_regressor.py ./{}/snapshot.py'.format(ckpt_path)
    else:
        cmd = r'copy train_acos_regressor.py {}\snapshot.py'.format(ckpt_path)
    print(cmd)
    os.system(cmd)
    
    file = open('{}/validation.txt'.format(ckpt_path), 'w')

    trans = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)

    while batch_num <= max_batch_num:
        batch_num += 1
        print('Epoch %03d: training...' % batch_num)
        reg.train()
        for (i, data) in enumerate(dataloader):
            joints = torch.as_tensor(data['joints'], device=device).reshape(-1, 19, 3)
            thetas = torch.as_tensor(data['thetas'], device=device)
            betas = torch.as_tensor(data['betas'], device=device)
            
            pred_thetas = reg(joints)
            _, recon_joints = smpl(betas, pred_thetas, trans)
            loss_joints = loss_op(recon_joints, joints)
            optimizer.zero_grad()
            loss_joints.backward()
            optimizer.step()
            
            if i % 32 == 0:
                print('batch %04d: loss: %10.6f' % (i, loss_joints.data.item()))
        
        print('Validation: ')
        reg.eval()
        data = dataset.rand_val_batch()
        joints = torch.as_tensor(data['joints'], device=device).reshape(-1, 19, 3)
        thetas = torch.as_tensor(data['thetas'], device=device)
        betas = torch.as_tensor(data['betas'], device=device)
        with torch.no_grad():
            pred_thetas = reg(joints)
            _, recon_joints = smpl(betas, pred_thetas, trans)
            loss_joints = loss_op(recon_joints, joints)
            line = 'Validation: loss_theta: %10.6f' % loss_joints.data.item()
            print(line)
            file.write(line+'\n')
            scheduler.step(loss_joints)
           
        if batch_num % 1 == 0:
            print('Save models...')
            torch.save(reg.state_dict(), '%s/regressor_%03d.pth' % (ckpt_path, batch_num))
    file.close()
