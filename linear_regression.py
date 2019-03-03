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


if __name__ == '__main__':
    pose_size = 72
    beta_size = 10
    
    np.random.seed(9608)
    device = torch.device('cuda')
    model = SMPLModel(device=device, model_path = 'model_24_joints.pkl')
    
    dataset = Joint2SMPLDataset('train_dataset_24_joints_1.0.pickle', batch_size, fix_beta_zero=True)
    
    for i in range(pose_size):
        # TODO: estimate regression parameters with dataset samples.
        
        