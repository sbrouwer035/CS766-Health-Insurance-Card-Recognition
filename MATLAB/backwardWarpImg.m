function [mask, result_img] = backwardWarpImg(src_img, resultToSrc_H,...
    dest_canvas_width_height)

dest_canvas_width = dest_canvas_width_height(1);
dest_canvas_height = dest_canvas_width_height(2);

[xd,yd] = meshgrid(1:dest_canvas_width, 1:dest_canvas_height);
%display([xd(:)]);
src_pts = applyHomography(resultToSrc_H,[xd(:),yd(:)]);

for i=1:3
    
    src_interp = interp2(src_img(:,:,i),src_pts(:,1),src_pts(:,2));
    result_img(:,:,i) = reshape(src_interp,dest_canvas_height,dest_canvas_width);

end

result_img(isnan(result_img)) = 0;
mask = rgb2gray(result_img);

%display(mask);
end
