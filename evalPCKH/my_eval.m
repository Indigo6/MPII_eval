% Evaluate performance by comparing predictions to ground truth annotations.

%%% OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% IDs of prediction sets to include in results
PRED_IDS = [1, 2, 3, 4, 5, 6];
% Subset of the data that the predictions correspond to ('val' or 'train')
plotcurve = false;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

addpath ('eval')

fprintf('# MPII single-person pose evaluation script\n')

range = 0:0.01:0.5;

tableDir = './latex'; if (~exist(tableDir,'dir')), mkdir(tableDir); end
plotsDir = './plots'; if (~exist(plotsDir,'dir')), mkdir(plotsDir); end
tableTex = cell(length(PRED_IDS)+1,1);

% load ground truth
p = getExpParams(-1)
load([p.gtDir '/annolist_dataset_v12'], 'annolist');
load([p.gtDir '/mpii_human_pose_v1_u12'], 'RELEASE');
annolist_test = annolist(RELEASE.img_train == 0);
% evaluate on the "single person" subset only
single_person_test = RELEASE.single_person(RELEASE.img_train == 0);
% convert to annotation list with a single pose per entry
[annolist_test_flat, single_person_test_flat] = flatten_annolist(annolist_test,single_person_test);
% represent ground truth as a matrix 2x14xN_images
gt = annolist2matrix(annolist_test_flat(single_person_test_flat == 1));
% compute head size
headSize = getHeadSizeAll(annolist_test_flat(single_person_test_flat == 1));

pckAll = zeros(length(range),16,length(PRED_IDS));

for i = 1:length(PRED_IDS);
  % load predictions
  p = getExpParams(PRED_IDS(i));
  try
    load(p.predFilename, 'preds');
  catch
    preds = h5read(p.predFilename, '/preds');
  end
  
  if size(preds, 1) == 2
    preds = permute(preds, [3, 2, 1]);
  end
  
  % Check that there are the same number of predictions and ground truth
  % annotations. If this assertion fails, a likely cause is a mismatch in
  % subsets (eg predictions are for the training set but ground truth
  % annotations are for the validation set).
  fprintf('%d\n', length(preds))
  fprintf('%d\n', length(gt))
  assert(length(preds) == length(gt));

  pred_flat = annolist_test_flat(single_person_test_flat == 1);
  for idx = 1:length(preds);
    for pidx = 1:length(pred_flat(idx).annorect.annopoints.point);
      joint = pred_flat(idx).annorect.annopoints.point(pidx).id + 1;
      xy = preds(idx, joint, :);
      pred_flat(idx).annorect.annopoints.point(pidx).x = xy(1);
      pred_flat(idx).annorect.annopoints.point(pidx).y = xy(2);
    end
  end

  % pred = annolist2matrix(pred_flat(single_person_flat == 1));
  pred = annolist2matrix(pred_flat);
  
  % only gt is allowed to have NaN
  pred(isnan(pred)) = inf;

  % compute distance to ground truth joints
  dist = getDistPCKh(pred,gt,headSize);

  % compute PCKh
  pck = computePCK(dist,range);

  % plot results
  [row, header] = genTablePCK(pck(end,:),p.name);
  tableTex{1} = header;
  tableTex{i+1} = row;

  pckAll(:,:,i) = pck;

  auc = area_under_curve(scale01(range),pck(:,end));
  fprintf('%s, AUC: %1.1f\n',p.name,auc);
end

% Save results
fid = fopen([tableDir '/pckh.tex'],'wt');assert(fid ~= -1);
for i=1:length(tableTex),fprintf(fid,'%s\n',tableTex{i}); end; fclose(fid);

% plot curves
bSave = true;
if (plotcurve)
    plotCurveNew(squeeze(pckAll(:,end,:)),range,PRED_IDS,'PCKh total, MPII',[plotsDir '/pckh-total-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[1 6],:),2)),range,PRED_IDS,'PCKh ankle, MPII',[plotsDir '/pckh-ankle-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[2 5],:),2)),range,PRED_IDS,'PCKh knee, MPII',[plotsDir '/pckh-knee-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[3 4],:),2)),range,PRED_IDS,'PCKh hip, MPII',[plotsDir '/pckh-hip-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[7 12],:),2)),range,PRED_IDS,'PCKh wrist, MPII',[plotsDir '/pckh-wrist-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[8 11],:),2)),range,PRED_IDS,'PCKh elbow, MPII',[plotsDir '/pckh-elbow-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[9 10],:),2)),range,PRED_IDS,'PCKh shoulder, MPII',[plotsDir '/pckh-shoulder-mpii'],bSave,range(1:5:end));
    plotCurveNew(squeeze(mean(pckAll(:,[13 14],:),2)),range,PRED_IDS,'PCKh head, MPII',[plotsDir '/pckh-head-mpii'],bSave,range(1:5:end));
end

display('Done.')
