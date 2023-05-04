function runProject(varargin)
% runProject is the "main" interface that lists a set of 
% functions corresponding to the OCR project
%
% 
% Usage:
% runProject                      : list all the registered functions
% runProject('function_name')      : execute a specific test

% Settings to make sure images are displayed without borders
orig_imsetting = iptgetpref('ImshowBorder');
iptsetpref('ImshowBorder', 'tight');
temp1 = onCleanup(@()iptsetpref('ImshowBorder', orig_imsetting));

fun_handles = {@warpImageManual,@logoBasedSearch,@logoBasedSearch_PreOCR,@OCR,@cardImgExtraction,@ROI_Identification,@demoMATLABTricks};

% Call test harness
runTests(varargin, fun_handles);


%%
%Testing code for warping a given image into a predefined blank template
%by manually selecting the region
function warpImageManual()

boundary = im2double(imread('TemplateMaster.png')); 
source_img = im2double(imread('Medicare.png')); 

f1 = figure();
imshow(boundary);
[xs,ys] = ginput(4);
display([xs,ys]);
boundary_pts = [xs,ys];

f2 = figure();
imshow(source_img);
[xd,yd] = ginput(4);
display([xd,yd]);
source_pts = [xd,yd];

delete(f1);
delete(f2);

H_3x3 = computeHomography(source_pts, boundary_pts);

dest_template_width_height = [size(boundary, 2), size(boundary, 1)];

[mask, dest_img] = backwardWarpImg(source_img, inv(H_3x3), dest_template_width_height);
mask = ~mask;

result = boundary .* cat(3, mask, mask, mask) + dest_img;

imwrite(result, 'TemplateMedicare.png');

%%

%Use SIFT to quickly identify logo 
%testing logo based template identification

function logoBasedSearch()

imgs = imread('Logomedicare.png'); imgd = imread('Medicare.png');

impl = 'MATLAB';
[xs, xd] = genSIFTMatches(imgs, imgd, impl);
display(length([xs,xd]));
%showCorrespondence is modified to allow resizing of images and only
%compare images in gray scale
logo_SIFT_img = showCorrespondence(imgs, imgd, xs, xd);
figure, imshow(logo_SIFT_img);
imwrite(logo_SIFT_img, 'logobasedSIFTresult.png');

%%

%If SIFT and RANSAC for logo based matching
%this function warps Medicarewithbackgroup.png into the desired dimention
%The final image "NormalizedMedicare.png" is ready for OCR

function logoBasedSearch_PreOCR()

imgs = imread('Logomedicare.png'); imgd = imread('Medicarewithbackground.png');

impl = 'MATLAB';
[xs, xd] = genSIFTMatches(imgs, imgd, impl);

if (length([xs,xd])>75)
    
    img_template = imread('TemplateMedicare.png'); 
    impl = 'MATLAB';
    [xs1, xd1] = genSIFTMatches(imgd,img_template,impl);
    
    ransac_n = 1000; % Max number of iterations
    ransac_eps = 2; %Acceptable alignment error 

    [inliers_id, H_3x3] = runRANSAC(xs1, xd1, ransac_n, ransac_eps);

    after_img = showCorrespondence(imgd,img_template, xs1(inliers_id, :), xd1(inliers_id, :));
    imwrite(after_img, 'after_ransactest2.png');

    % Warp the imgd into image template
    boundary=im2double(imread('TemplateMaster.png'));
    imgd=im2double(imgd);
    dest_template_width_height = [size(boundary, 2), size(boundary, 1)];

    [mask, dest_img] = backwardWarpImg(imgd, inv(H_3x3), dest_template_width_height);
    % mask should be of the type logical
    mask = ~mask;
    % Superimpose the image
    result = boundary .* cat(3, mask, mask, mask) + dest_img;
    %figure, imshow(result);
    imwrite(result, 'NormalizedMedicare.png');

end

%%

%text recognition
%result image: PostOCRMedicare.png

function OCR()
norm_img=imread('NormalizedMedicare.png');
%norm_img = imsharpen(imadjust(rgb2gray(imread('NormalizedMedicare.png'))));
threshold = graythresh(norm_img);
norm_img = im2bw(norm_img,0.5);
display(threshold);
f1 = figure();
imshow(norm_img);
%[xs,ys] = ginput(2);
%display([xs,ys]);
%display(size(norm_img));
%roi = round(getPosition(drawrectangle));
imwrite(norm_img,'PreOCRMedicare.png');
norm_img=im2double(norm_img);

roi = [550 340 220 55];
ocrResults = ocr(norm_img,roi);
Iocr = insertObjectAnnotation(norm_img,"rectangle", ...
            ocrResults.WordBoundingBoxes,("Sex: " + ocrResults.Words),LineWidth=5,FontSize=40);

roi = [30 340 420 55];
ocrResults = ocr(norm_img,roi);
Iocr = insertObjectAnnotation(Iocr,"rectangle", ...
            ocrResults.WordBoundingBoxes,("ID: " + ocrResults.Words),LineWidth=5,FontSize=40);

roi = [30 250 500 55];
ocrResults = ocr(norm_img,roi);
Iocr = insertObjectAnnotation(Iocr,"rectangle", ...
            ocrResults.WordBoundingBoxes,("Name: " + ocrResults.Words),LineWidth=5,FontSize=40);
%figure
%imshow(Iocr);
%display(ocrResults);
imwrite(Iocr,'PostOCRMedicare.png');
delete(f1);
    

%%
% Card dimension 8.5 X 5.4
% Extract the card region from a given image.
% Input images: 'cardExtractionTestImg1.jpg; cardExtractionTestImg2.jpg; Medicarewithbackground.png'

function cardImgExtraction()

I = imread('cardExtractionTestImg1.jpg'); %Input images: 'cardExtractionTestImg1.jpg; cardExtractionTestImg2.jpg; Medicarewithbackground.png'

rotI = imrotate(I,0,'crop');
I = rgb2gray(I);
BW = edge(I,'canny',0.3);
[H,T,R] = hough(BW);
imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
P  = houghpeaks(H,5,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));
plot(x,y,'s','color','white');

lines = houghlines(BW,T,R,P,'FillGap',100,'MinLength',7);
figure, imshow(rotI), hold on
max_len = 0;

line_traits = zeros([length(lines),6]);  % c1 is length, c2 is abs angle,c3 is a, c4 is b for y=ax+b,c5 is line theta, c6 is line number

for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   line_traits(k,1) = distance(lines(k).point1,lines(k).point2);


   if lines(k).theta > 0
        line_traits(k,2) =lines(k).theta-90;
   else
        line_traits(k,2) = lines(k).theta+90;
   end
   
      
   if abs(lines(k).theta) >1
        coefficients = polyfit([lines(k).point1(1),lines(k).point2(1)], [lines(k).point1(2),lines(k).point2(2)], 1);
   
        line_traits(k,3) = coefficients (1);
        line_traits(k,4) = coefficients (2);
   else
        line_traits(k,4) = Inf; % a very big number
   end
   
   line_traits(k,5)=lines(k).theta;
   line_traits(k,6)=k;
   %display(line_traits(k,:));

   %if abs(lines(k).theta)>=0
   %     plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
   %     %display(lines(k));
        % Plot beginnings and ends of lines
   %     plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');
   %     plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');
   %end

   % Determine the endpoints of the longest line segment 
   len = norm(lines(k).point1 - lines(k).point2);
   if ( len > max_len)
      max_len = len;
      xy_long = xy;
      max_k = k;
      %display(lines(k));
   end
end

long_edge_theta = lines(max_k).theta; %use column 6 of line traits
short_edge_theta = line_traits(max_k,2); % use column 2 of line traits

if abs(long_edge_theta) > 45
    long_edge_indices = find ((line_traits(:,5) > long_edge_theta-2) & (line_traits(:,5) < long_edge_theta+2));
    
    %find the top edge and bottom edge and their distance (card width)
    % card width /5.4 * 8.5 = card length

    long_edge_top=long_edge_indices(1);
    long_edge_bottom=long_edge_indices(1);
    
    for i=1:1:size(long_edge_indices)
        if line_traits(long_edge_indices(i),4) < line_traits(long_edge_top,4)
        long_edge_top=long_edge_indices(i);
        end

        if line_traits(long_edge_indices(i),4) > line_traits(long_edge_bottom,4)
        long_edge_bottom=long_edge_indices(i);
        end
    end

    card_width = abs((line_traits(long_edge_top,4) - line_traits(long_edge_bottom,4)) * sin(long_edge_theta * pi / 180)); %add padding
    card_length = card_width / 5.4 * 8.5;
    %display(card_length);

    %test
    %xy = [lines(long_edge_bottom).point1; lines(long_edge_bottom).point2];
    %plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
    %

    short_edge_indices = find((line_traits(:,5) > short_edge_theta-2) & (line_traits(:,5) < short_edge_theta+2));
    short_edge_left=short_edge_indices(1);

    for i=1:1:size(short_edge_indices)
        
        if lines(short_edge_indices(i)).point1(1) < lines(short_edge_left).point1(1)
            short_edge_left=short_edge_indices(i);
        end
        
    end

    
    %test
    %display(short_edge_left);
    %xy = [lines(short_edge_left).point1; lines(short_edge_left).point2];
    %plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');
    %
    %xy = [lines(8).point1; lines(8).point2];
    %plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','green');

        
    edge_points=zeros(4,2);
    
    %find intersection left top

    if abs(line_traits(short_edge_left,5)) < 2
        edge_points(1,1) = lines(short_edge_left).point1(1);
        edge_points(1,2) = line_traits(long_edge_top,3) * edge_points(1,1) + line_traits(long_edge_top,4);
        edge_points(2,1) = edge_points(1,1) + card_length;
        edge_points(2,2) = line_traits(long_edge_top,3) * edge_points(2,1) + line_traits(long_edge_top,4);
        edge_points(3,1) = edge_points(2,1);
        edge_points(3,2) = line_traits(long_edge_bottom,3) * edge_points(3,1) + line_traits(long_edge_bottom,4);
        edge_points(4,1) = edge_points(1,1);
        edge_points(4,2) = line_traits(long_edge_bottom,3) * edge_points(3,1) + line_traits(long_edge_bottom,4);
    else    
        edge_points(1,1) = -(line_traits(long_edge_top,4)-line_traits(short_edge_left,4))/(line_traits(long_edge_top,3)-line_traits(short_edge_left,3));
        edge_points(1,2) = line_traits(long_edge_top,3) * edge_points(1,1) + line_traits(long_edge_top,4);
        edge_points(2,1) = edge_points(1,1) + card_length*(-sin(long_edge_theta * pi / 180));
        edge_points(2,2) = line_traits(long_edge_top,3) * edge_points(2,1) + line_traits(long_edge_top,4);
        edge_points(4,1) = -(line_traits(long_edge_bottom,4)-line_traits(short_edge_left,4))/(line_traits(long_edge_bottom,3)-line_traits(short_edge_left,3));
        edge_points(4,2) = line_traits(long_edge_bottom,3) * edge_points(4,1) + line_traits(long_edge_bottom,4);
        edge_points(3,1) = edge_points(4,1) + card_length*(-sin(long_edge_theta * pi / 180));
        edge_points(3,2) = line_traits(long_edge_bottom,3) * edge_points(3,1) + line_traits(long_edge_bottom,4);
    
    end
    
    plot([edge_points(1,1);edge_points(2,1)],[edge_points(1,2);edge_points(2,2)],'LineWidth',3,'Color','red');
    plot([edge_points(2,1);edge_points(3,1)],[edge_points(2,2);edge_points(3,2)],'LineWidth',3,'Color','blue');
    plot([edge_points(3,1);edge_points(4,1)],[edge_points(3,2);edge_points(4,2)],'LineWidth',3,'Color','green');
    plot([edge_points(1,1);edge_points(4,1)],[edge_points(1,2);edge_points(4,2)],'LineWidth',4,'Color','black');
    
    %display(x)
    %plot(edge_points,'x','LineWidth',2,'Color','red');
    
end



%%

% Diff -> cleanup -> region ID 
% Given 2 sample images, the sample program will find the region containing
% the variable data points. Classify the ROI and produce the instruction
% file

function ROI_Identification()

    img_1= imread('ROI_Image_Sample1.png');
    img_2= imread('ROI_Image_Sample2.png');

    %Img Diff
    K = imabsdiff(img_1,img_2);
    
    %Cleanup
    BW = imbinarize(K);
    BW = imfill(BW,8,"holes");
    BW = bwareaopen(BW,200);
    
    figure
    imshowpair(K,BW,'montage')

    
    %ROI Determination
    [y,x] = find(BW);
    coord = [x, y];
    
    min_x=min(coord(:,1));
    max_x=max(coord(:,1));
    min_y=min(coord(:,2));
    max_y=max(coord(:,2));
    %display([min_x min_y max_x max_y]);

    %imshow(img_1);
    roi = [min_x min_y 500 max_y-min_y];
    %rectangle('Position',roi,'LineWidth',3,'LineStyle','-')
    img_1=im2double(img_1);
    ocrResults = ocr(img_1,roi);
    Iocr = insertObjectAnnotation(img_1,"rectangle", ...
            ocrResults.WordBoundingBoxes,(" " + ocrResults.Words),LineWidth=5,FontSize=40);
    
    imwrite(Iocr,'ROI_Identification_withOCR.png');
    
    %compare the OCR result to EMR Database, then classify the ROI and
    %Create Instruction file : using the classified ROI
    fid = fopen( 'InstructionMedicare.txt', 'wt' );
        fprintf( fid, '%s,%d,%d,%d,%d','Subscriber Name: ',roi(1,1),roi(1,2),roi(1,3),roi(1,4));
    fclose(fid);
    %Create Template file: Image - ROI




    


