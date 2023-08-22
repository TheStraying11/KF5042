imds = imageDatastore('breastCancerDataSet', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); % get all images

files = imds.Files; % get filenames

for i = 1 : length(files)
    curr = files{i};

    if contains(curr, 'mask')
        delete(curr) % we dont want the masks
    end
end