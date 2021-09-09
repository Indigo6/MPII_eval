function p = getExpParams(predidx)

% path to the directory containg ground truth 'annolist' and RELEASE structures
p.gtDir = '../ground_truth/';
p.predFilename = '../data/pred_tv_0.8.mat';
p.name = 'test';
p.colorIdxs = [1 1];
p.partNames = {'right ankle','right knee','right hip','left hip','left knee','left ankle','right wrist','right elbow','right shoulder','left shoulder','left elbow','left wrist','neck','top head','avg full body'};
switch predidx
    case 0
	% replicated RELEASE structure containing predictions on test images only
	% predictions are stored in the same way as GT body joint annotations
        % i.e. annolist_test = annolist(RELEASE.img_train == 0);
        p.predFilename = '../data/pred.mat';
        p.colorIdxs = [7 1];
end

p.colorName = getColor(p.colorIdxs);
p.colorName = p.colorName ./ 255;

end
