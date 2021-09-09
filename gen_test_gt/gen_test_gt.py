import json
import numpy as np

from numpy import array
from scipy.io import loadmat, savemat


def save_joints_json(mat1):
    joint_data_fn = 'test.json'
    total_json = []
    for i in range(mat1['test'].shape[1]):
        anno = mat1['test']['annorect'][0,i]
        img_fn = mat1['test']['image'][0,i][0,0][0][0]

        anno_fields = str(anno.dtype)
        if 'objpos' not in anno_fields or 'scale' not in anno_fields or\
            'annopoints' not in anno_fields:
            print(i)
            continue

        temp_json = {}
        temp_json['image'] = img_fn
        temp_json['center'] = [round(anno['objpos'][0,0]['x'][0,0][0][0],2),
                            round(anno['objpos'][0,0]['y'][0,0][0][0],2)]
        temp_json['scale'] = round(float(anno['scale'][0,0][0,0]), 8)

        x1 = round(float(anno['x1'][0,0][0,0]))
        x2 = round(float(anno['x2'][0,0][0,0]))
        y1 = round(float(anno['y1'][0,0][0,0]))
        y2 = round(float(anno['y2'][0,0][0,0]))
        temp_json['head'] = [[x1, y1], [x2, y2]]

        joints_vis = [0.0] * 16
        joints = [[0.0, 0.0]] * 16

        for j in range(anno['annopoints'][0,0][0,0][0]['x'].shape[1]):
            joint_x = round(float(anno['annopoints'][0,0][0,0][0]['x'][0,j][0,0]), 2)
            joint_y = round(float(anno['annopoints'][0,0][0,0][0]['y'][0,j][0,0]), 2)
            joint_id = anno['annopoints'][0,0][0,0][0]['id'][0,j][0,0]
            joint_vis = anno['annopoints'][0,0][0,0][0]['is_visible'][0,j]
            joints_vis[joint_id] = float(0) if joint_vis.size == 0 else float(joint_vis[0][0])
            joints[joint_id] = [joint_x, joint_y]

        temp_json['joints'] = joints
        temp_json['joints_vis'] = joints_vis
        total_json.append(temp_json)

    # print(count)
    with open(joint_data_fn, "w", encoding='utf-8') as f:
        # indent 超级好用，格式化保存字典，默认为None，小于0为零个空格
        # f.write(json.dumps(total_json, indent=4))
        json.dump(total_json, f, indent=4)   # 和上面的效果一样


def save_joints_mat(mat1):
    test_dict = {} 
    test_dict['dataset_joints'] = array([[array(['rank'], dtype='<U4'), array(['rkne'], dtype='<U4'),
                                        array(['rhip'], dtype='<U4'), array(['lhip'], dtype='<U4'),
                                        array(['lkne'], dtype='<U4'), array(['lank'], dtype='<U4'),
                                        array(['pelv'], dtype='<U4'), array(['thor'], dtype='<U4'),
                                        array(['neck'], dtype='<U4'), array(['head'], dtype='<U4'),
                                        array(['rwri'], dtype='<U4'), array(['relb'], dtype='<U4'),
                                        array(['rsho'], dtype='<U4'), array(['lsho'], dtype='<U4'),
                                        array(['lelb'], dtype='<U4'), array(['lwri'], dtype='<U4')]],
                                        dtype=object)

    pos_gt_src = np.zeros((7247, 16, 2), dtype=float)
    jnt_vis = np.zeros((7247, 16), dtype=float)
    headboxes_src = np.zeros((7247, 2, 2), dtype=float)

    idx = -1
    for i in range(mat1['test'].shape[1]):
        anno = mat1['test']['annorect'][0,i]
        img_fn = mat1['test']['image'][0,i][0,0][0][0]

        anno_fields = str(anno.dtype)
        if 'objpos' not in anno_fields or 'scale' not in anno_fields or\
            'annopoints' not in anno_fields:
            print(i)
            continue
        
        x1 = round(float(anno['x1'][0,0][0,0]))
        x2 = round(float(anno['x2'][0,0][0,0]))
        y1 = round(float(anno['y1'][0,0][0,0]))
        y2 = round(float(anno['y2'][0,0][0,0]))
        headboxes_src[i,0,0] = x1
        headboxes_src[i,0,1] = y1
        headboxes_src[i,1,0] = x2
        headboxes_src[i,1,1] = y2

        for j in range(anno['annopoints'][0,0][0,0][0]['x'].shape[1]):
            joint_x = anno['annopoints'][0,0][0,0][0]['x'][0,j][0,0]
            joint_y = anno['annopoints'][0,0][0,0][0]['y'][0,j][0,0]
            joint_id = anno['annopoints'][0,0][0,0][0]['id'][0,j][0,0]
            joint_vis = anno['annopoints'][0,0][0,0][0]['is_visible'][0,j]
            joint_vis = 0 if joint_vis.size == 0 else joint_vis[0,0]
            pos_gt_src[i,joint_id,0] = joint_x
            pos_gt_src[i,joint_id,1] = joint_y
            jnt_vis[i,joint_id] = joint_vis

                
    print(i)
    jnt_missing = 1-jnt_vis
    test_dict['headboxes_src'] = headboxes_src.transpose(1, 2, 0)
    test_dict['pos_gt_src'] = pos_gt_src.transpose(1, 2, 0)
    test_dict['jnt_missing'] = jnt_missing.transpose(1, 0)

    savemat('gt_test.mat', mdict=test_dict)


if __name__ == '__main__':
    mat = loadmat('../data/test.mat')
    save_joints_json(mat)
    save_joints_mat(mat)