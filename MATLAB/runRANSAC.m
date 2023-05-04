function [inliers_id, H] = runRANSAC(Xs, Xd, ransac_n, eps)

max=0;

for i=1:ransac_n

    rand_index = randi(length(Xs),1,4);
    H_temp = computeHomography(Xs(rand_index,:),Xd(rand_index,:));
    X_temp = applyHomography(H_temp,Xs);
    Diff = X_temp - Xd;

    %S = sum(A,dim) returns the sum along dimension dim. For example, 
    % if A is a matrix, then sum(A,2) is a column vector containing the sum of each row.
    dist_euclidean = sqrt(sum(Diff .* Diff,2)); 
    %display(dist_euclidean);
    
    ransac_inlier =  find(dist_euclidean < eps); %bigger eps, more points, less accurate 
    
    %after ransac_n iterations, we are going to find the H where max (out of ransac_n iterations) number
    %of estimated points (X_temp) falls within eps distance of target points in Xd
    if (length(ransac_inlier) > max)
        max = length(ransac_inlier);
        H = H_temp;
        inliers_id = ransac_inlier;
    end

end
end