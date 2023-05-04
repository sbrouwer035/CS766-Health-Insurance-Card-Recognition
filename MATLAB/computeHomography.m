function H_3x3 = computeHomography(src_pts_nx2, dest_pts_nx2)

%LEC 10 pg 43

n = size(src_pts_nx2,1);
A = zeros(2 * n , 9); 

for i = 1:n

    xs = src_pts_nx2(i,1);
    ys = src_pts_nx2(i,2);
    xd = dest_pts_nx2(i,1);
    yd = dest_pts_nx2(i,2);

    A(2*i-1,:) = [xs ys 1 0 0 0 -xd*xs -xd*ys -xd];
    A(2*i,:) = [0 0 0 xs ys 1 -yd*xs -yd*ys -yd];

end


%[V,D] = eig(A) returns diagonal matrix D of eigenvalues 
% and matrix V whose columns are the corresponding right eigenvectors, 
% so that A*V = V*D.

[V,~] = eig(A'*A);
%display(V);
%display(D);

H_3x3=(reshape(V(:,1),3,3))';
%display(H_3x3);

end