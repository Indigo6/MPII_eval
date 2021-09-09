function gen_test_flat_mat()

fprintf('gen_test_flat_mat()\n');

load('../ground_truth/annolist_dataset_v12','annolist');
load('../ground_truth/mpii_human_pose_v1_u12','RELEASE');
annolist = annolist(RELEASE.img_train == 0);
single_person = RELEASE.single_person(RELEASE.img_train == 0);
image_list = RELEASE.annolist(RELEASE.img_train == 0);

annolist_flat = struct('image',[],'annorect',[]);

n = 0;
single_person_flat = [];
for imgidx = 1:length(annolist)
    rect_gt = annolist(imgidx).annorect;
    for ridx = 1:length(rect_gt)
        if (isfield(rect_gt(ridx),'objpos') && ~isempty(rect_gt(ridx).objpos))
            n = n + 1;
            annolist_flat(n).image.name = image_list(imgidx).image.name;
            annolist_flat(n).annorect = rect_gt(ridx);
            single_person_flat(n) = ismember(ridx,single_person{imgidx});
        end
    end
end

test = annolist_flat(single_person_flat == 1);
save('../data/test.mat','test')

end