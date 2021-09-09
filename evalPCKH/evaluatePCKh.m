function evaluatePCKh()

fprintf('genPreds()\n');

% load ground truth
load('../ground_truth/annolist_dataset_v12','annolist');
load('../ground_truth/mpii_human_pose_v1_u12','RELEASE');
load('../data/pred_7_dark.mat','preds');

annolist = annolist(RELEASE.img_train == 0);
single_person = RELEASE.single_person(RELEASE.img_train == 0);

pred = annolist;
i = 0;
for imgidx = 1:length(annolist)
    rect_gt = annolist(imgidx).annorect;
    for ridx = 1:length(rect_gt)
        if (isfield(rect_gt(ridx),'objpos') && ~isempty(rect_gt(ridx).objpos) && ismember(ridx,single_person{imgidx}))
            i = i + 1;
            temp_pred = preds(i,:,:);
            for pidx = 1:length(temp_pred)
                for j = 1:length(pred(imgidx).annorect(ridx).annopoints.point)
                    if pred(imgidx).annorect(ridx).annopoints.point(j).id == pidx - 1
%                         fprintf('imgidx=%d, j=%d,pidx=%d\n',imgidx,j,pidx)
                        pred(imgidx).annorect(ridx).annopoints.point(j).x = temp_pred(1,pidx,1);
                        pred(imgidx).annorect(ridx).annopoints.point(j).y = temp_pred(1,pidx,2);
                        break;
                    end
                end
            end
        end
    end
end
fprintf('i=%d\n',i);
save('../data/pred_keypoints_mpii.mat','pred');

% implementation of PCKh measure,
% as defined in [Andriluka et al., CVPR'14]

fprintf('evaluatePCKh()\n');


range = 0:0.01:0.5;

% load ground truth
load('../ground_truth/annolist_dataset_v12','annolist');
load('../ground_truth/mpii_human_pose_v1_u12','RELEASE');
annolist_test = annolist(RELEASE.img_train == 0);
% evaluate on the "single person" subset only
single_person_test = RELEASE.single_person(RELEASE.img_train == 0);
% convert to annotation list with a single pose per entry
[annolist_test_flat, single_person_test_flat] = flatten_annolist(annolist_test,single_person_test);
% represent ground truth as a matrix 2x14xN_images
gt = annolist2matrix(annolist_test_flat(single_person_test_flat == 1));
% compute head size
headSize = getHeadSizeAll(annolist_test_flat(single_person_test_flat == 1));


% load predictions
load('../data/pred_keypoints_mpii.mat','pred');
assert(length(annolist_test) == length(pred));
pred_flat = flatten_annolist(pred,single_person_test);
pred = annolist2matrix(pred_flat(single_person_test_flat == 1));
% only gt is allowed to have NaN
pred(isnan(pred)) = inf;

% compute distance to ground truth joints
dist = getDistPCKh(pred,gt,headSize);

% compute PCKh
pck = computePCK(dist,range);

% plot results
[row, header] = genTablePCK(pck(end,:),'ours');

auc = area_under_curve(scale01(range),pck(:,end));
fprintf('%s, AUC: %1.1f\n','ours',auc);

end