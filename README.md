### 生成JSON格式的标注

1. 使用 `gen_test_flat_mat/gen_test_flat_mat.m` 生成摊平的 test 数据集标注信息 `test.mat`
2. 使用 `gen_test_gt/gen_test_gt.ipynb` 或者   `gen_test_gt/gen_test_gt.py` 生成 `gt_test.mat` 和 `test.json` 
3. 将 `gt_test.mat` 和 `test.json` 复制到 HRNet 的 `data\mpii\annot` 目录下，

### 测试

1. 进行 `DATASET.TEST_SET test` 的测试，并将 `output` 相应文件夹生成的 `pred.mat` 复制到 此项目的 `data` 目录下
2. 使用 `gen_preds/genPreds.m` 生成 `pred_keypoints_mpii.mat` 
3. 最后使用 `evalPCKH/evaluatePCKh.m` 生成测试结果

