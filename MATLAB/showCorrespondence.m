function result_img = ...
    showCorrespondence(orig_img, warped_img, src_pts_nx2, dest_pts_nx2)

f1=figure();

orig_img=rgb2gray(orig_img);
warped_img=rgb2gray(warped_img);
[orig_y,orig_x] = size(orig_img);
[warped_y,warped_x] = size(warped_img);

diff=abs(warped_y-orig_y);
if (orig_y < warped_y)
    orig_img = [orig_img; zeros(diff,orig_x)];
elseif (warped_y < orig_y)
    warped_img = [warped_img; zeros(diff,warped_x)];
end

[orig_y,orig_x] = size(orig_img);
[warped_y,warped_x] = size(warped_img);


diff = abs(warped_x-orig_x);
if (orig_x < warped_x)
    orig_img = [orig_img,zeros(orig_y,diff)];
elseif (warped_x < orig_x)
    warped_img = [warped_img, zeros(warped_y,diff)];
end


imshow([orig_img,warped_img]);
hold on;
for i = 1:length(src_pts_nx2)
    line([src_pts_nx2(i,1), dest_pts_nx2(i,1) + size(orig_img,2)], [src_pts_nx2(i,2), dest_pts_nx2(i,2)],'LineWidth',1.5);
end
hold off; 

result_img = saveAnnotatedImg(f1);
delete(f1);


end
