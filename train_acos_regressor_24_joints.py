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
    def __init__(self, hidden_dim=256, indim=72, thetadim=72, betadim=10,
                batch_size=64, hidden_layer=3, use_dropout=False):
        super(AcosRegressor, self).__init__()
        self.limbs_index = torch.tensor([
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24],
            [2, 13, 4, 5, 6, 7, 13, 13, 8, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 16, 21, 22, 23]
            #[1, 2, 8, 9, 3, 4, 7, 8, 12, 12, 9, 10, 14, 14, 13, 13, 15, 16]
        ], dtype=torch.long)
        self.limbs_index -= torch.ones_like(self.limbs_index)  # convert to 0-index
        
        model = [ResBlock1d(indim=23*23+24*3, outdim=hidden_dim)]
        for i in range(hidden_layer):
            model += [ResBlock1d(indim=hidden_dim, use_dropout=use_dropout)]
        model += [nn.Linear(hidden_dim, thetadim)]
        self.model = nn.Sequential(*model)
        self.clamp_eps = 1e-6
        self.norm_eps = 1e-9
        
    def forward(self, x):
        # expect N * 19 * 3
        vec = x[:, self.limbs_index[0], :] - x[:, self.limbs_index[0], :]
        # 20190220: normalize vector!!!
        norm_vec = torch.norm(vec, dim=2, keepdim=True) + self.norm_eps
        vec /= norm_vec
        prod = torch.bmm(vec, vec.transpose(1, 2))
        # 20190220；clamp input to avoid NaN
        prod = torch.clamp(prod, min=(-1+self.clamp_eps), max=1-self.clamp_eps)
        if torch.isnan(prod).any():
            print('prod nan')
        angles = torch.acos(prod).view(-1, 23*23)
        if torch.isnan(angles).any():
            print('angles nan')
        # 20190301: Only use bone vectors and angles (Bad)
        features = torch.cat((x.view(-1, 24*3), angles), dim=1)
        return self.model(features)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    torch.backends.cudnn.enabled=True
    batch_size = 64
    max_batch_num = 40
    
    #dataset = Joint2SMPLDataset('train_dataset.pickle', batch_size)
    theta_var = 1.0
    training_stage = 5
    dataset = Joint2SMPLDataset('train_dataset_24_joints_1.0.pickle', batch_size, fix_beta_zero=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    torch.set_default_dtype(torch.float64)
    device = torch.device('cuda')
    reg = AcosRegressor(batch_size=batch_size).cuda()
    smpl = SMPLModel(device=device,
        model_path = './model_24_joints.pkl',
                    simplify=True
    )
    loss_op = nn.L1Loss()
    
    optimizer = optim.Adam(reg.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.25, patience=1, verbose=True)
    
    batch_num = 0
    ckpt_path = 'checkpoints_0303_24_joints'.format(theta_var)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    if batch_num > 0 and os.path.isfile('%s/regressor_%03d.pth' % (ckpt_path, batch_num)):
        state_dict = torch.load_state_dict('%s/regressor_%03d.pth' % (ckpt_path, batch_num))
        reg.load(state_dict)

    # copy current file into checkpoint folder to record parameters, ugly.
    if platform == 'linux':
        cmd = 'cp train_acos_regressor_24_joints.py ./{}/snapshot.py'.format(ckpt_path)
    else:
        cmd = r'copy train_acos_regressor_24_joints.py {}\snapshot.py'.format(ckpt_path)
    print(cmd)
    os.system(cmd)
    
    file = open('{}/validation.txt'.format(ckpt_path), 'w')

    trans = torch.zeros((batch_size, 3), dtype=torch.float64, device=device)

    while batch_num < max_batch_num:
        batch_num += 1
        print('Epoch %03d: training...' % batch_num)
        reg.train()
        for (i, data) in enumerate(dataloader):
            joints = torch.as_tensor(data['joints'], device=device)
            thetas = torch.as_tensor(data['thetas'], device=device)
            betas = torch.as_tensor(data['betas'], device=device)
            
            pred_thetas = reg(joints)
            _, recon_joints = smpl(betas, pred_thetas, trans)
            loss_joints = loss_op(recon_joints, joints)
            loss_thetas = loss_(pred_thetas, thetas)
            loss = loss_thetas + 5 * loss_joints
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i % 32 == 0:
                print('batch %04d: loss joints: %10.6f loss thetas: % 10.6f' \
                    % (i, loss_joints.data.item(), loss_thetas.data.item()))
        
        print('Validation: ')
        reg.eval()
        data = dataset.rand_val_batch()
        joints = torch.as_tensor(data['joints'], device=device)
        thetas = torch.as_tensor(data['thetas'], device=device)
        betas = torch.as_tensor(data['betas'], device=device)
        with torch.no_grad():
            pred_thetas = reg(joints)
            _, recon_joints = smpl(betas, pred_thetas, trans)
            loss_joints = loss_op(recon_joints, joints)
            loss_thetas = loss_op(pred_thetas, thetas)
            line = 'batch %04d: loss joints: %10.6f loss thetas: % 10.6f' \
                    % (i, loss_joints.data.item(), loss_thetas.data.item())
            print(line)
            file.write(line+'\n')
            scheduler.step(loss_joints)
           
        if batch_num % 5 == 0:
            print('Save models...')
            torch.save(reg.state_dict(), '%s/regressor_%03d.pth' % (ckpt_path, batch_num))
            '''
            if batch_num % 20 == 0 and training_stage < 5:
                # Fine-tuning on the next dataset with larger theta_var
                line = 'Switching dataset from theta_var = {}'.format(theta_var)
                theta_var += 0.2
                training_stage += 1
                dataset = Joint2SMPLDataset('train_dataset_{}.pickle'.format(training_stage), 
                    batch_size, fix_beta_zero=True)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
                # Renew optimizer and scheduler
                optimizer = optim.Adam(reg.parameters(), lr=0.0005, betas=(0.5, 0.999), weight_decay=1e-4)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                    factor=0.25, patience=1, verbose=True)
                line += ' to theta_var = {}\n'.format(theta_var)
                file.write(line)
                print(line)
             '''
    
    file.close()
