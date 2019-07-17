function compute_bird_view_orig ()
D = 'to_rectified';
filePattern = fullfile(D, '*.jpg');
pngFiles = dir(filePattern);
path = 'rectified';

for k = 1:length(pngFiles) 

  baseFileName = pngFiles(k).name;
  fullFileName = fullfile(D, baseFileName); 
  img_in = imread(fullFileName);
  img_in = imresize(img_in, [1024, 2048], 'nearest');
  
  focalLength = [2262.52 2265.3017905988554];
  principalPoint = [1096.98 513.137];
  imageSize = [size(img_in,1) size(img_in,2)];

  camIntrinsics = cameraIntrinsics(focalLength,principalPoint,imageSize);

  height = 1.7;
  pitch = 0.038*180.0/pi;
  yaw = -0.0195*180/pi;
  roll = 0;

  sensor = monoCamera(camIntrinsics,height,'Pitch',pitch, 'Yaw', yaw, 'Roll', roll);

  distAhead = 25;
  spaceToOneSide = 11;
  bottomOffset = 9.1;

  outView = [bottomOffset,distAhead,-spaceToOneSide,spaceToOneSide];
  outImageSize = [NaN size(img_in,2)];

  birdsEye = birdsEyeView(sensor,outView,outImageSize);
  img_out{k} = transformImage(birdsEye,img_in);
  
  %psf=fspecial('motion',2,162);
  %img_out{k}=deconvwnr(img_out{k},psf);
  app = k+99;
  imgName = strcat(num2str(app),'.jpg');
  fullFileName = fullfile(path, imgName); 
  imwrite(img_out{k}, fullFileName); % img respresents input image.

end

%for k = 1:numel(pngFiles)
  %  figure; imshow(img_out{k});
%end

end