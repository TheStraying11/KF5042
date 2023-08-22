lgraph = darknet53("Weights","none");
inputSize = [256, 256, 3]
imds = imageDatastore('breastCancerDataSet', ... % load the dataset
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7); % use 70% of it for training, and 30% for validation

[learnableLayer, classLayer] = findLayersToReplace(lgraph); % from matlab examples

numClasses = numel(categories(imdsTrain.Labels)); % get class labels

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer') % create new fully connected layer
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer); % from matlab examples, replace the fullyConnectedLayer with a new one

newClassLayer = classificationLayer('Name','new_classoutput'); % create new classLayer
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer); % from matlab examples, replace the classLayer with a new one

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10)); % freeze the layers inbetween
lgraph = createLgraphUsingConnections(layers,connections);

pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ... % randomize the data slightly to avoid bias
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation); % resize the new augmented data to fit the network

miniBatchSize = 10;
valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ... % set the options to train the network
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',60, ... % experiment variable 1
    'InitialLearnRate',3e-5, ... % experiment variable 2
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(augimdsTrain,lgraph,options); % train the network

[YPred,probs] = classify(net,augimdsValidation); % predict using the test data

accuracy = mean(YPred == imdsValidation.Labels); % calculate the accuracy of the model

print(accuracy)

