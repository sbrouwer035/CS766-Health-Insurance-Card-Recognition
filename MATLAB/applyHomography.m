function dest_pts_nx2 = applyHomography(H_3x3, src_pts_nx2)


%display(src_pts_nx2);
source_Matrix = [src_pts_nx2' ; ones(1,length(src_pts_nx2))];

%display(Source_Matrix);

dest_Matrix = H_3x3 * source_Matrix;

%display(dest_Matrix);

for i=1:2
    dest_matrix_norm(i,:) = dest_Matrix(i,:) ./ dest_Matrix(3,:);
end

dest_pts_nx2 = dest_matrix_norm';

%display(src_pts_nx2);
%display(dest_pts_nx2);
end
