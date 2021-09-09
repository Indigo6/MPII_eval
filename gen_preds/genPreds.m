function genPreds()

fprintf('genPreds()\n');

% load ground truth
load('../ground_truth/annolist_dataset_v12','annolist');
load('../ground_truth/mpii_human_pose_v1_u12','RELEASE');
load('../data/pred_srgl_tv_0.8.mat','preds');

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
end