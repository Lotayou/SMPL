import numpy as np
from scipy.sparse import coo_matrix
import pickle

if __name__ == '__main__':
    ids = np.array([
    3052,	460,
    3060,	3012,
    5459,	5537,
    5386,	5440,
    5218,	5126,
    5304,	6406,
    4084,	6429,
    595	,   2966,
    605	,   2999,
    1749,	1659,
    1926,	1938,
    2000,	2050,
    3076,	3014,
    1329,	3173,
    3500,	3022,
    1806,	3484,
    4952,	4325,
    4533,	4535,
    6734,	6730,
    6765,	6839,
    1479,	837, 
    1047,	1050,
    3334,	3331,
    3365,	3439
    ], dtype=np.int).reshape(24,2)
    dense = np.zeros((24, 6890), dtype=np.float64)
    for i in range(24):
        dense[i, ids[i][0]] = 0.5
        dense[i, ids[i][1]] = 0.5
        
    _24_joint_regressor = coo_matrix(dense)
    
    with open('./model.pkl', 'rb') as rf:
        params = pickle.load(rf)

    params['joint_regressor'] = _24_joint_regressor
    
    with open('./model_24_joints.pkl', 'wb') as wf:
        pickle.dump(params, wf)
    
