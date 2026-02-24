clc; clear; close all;                      % Komut penceresini temizle, değişkenleri sil, figürleri kapat
parallel.gpu.enableCUDAForwardCompatibility(true); % CUDA ileri uyumluluğunu aç

%% Dataset yolu
dataPath = 'C:\Users\LENOVO\Desktop\MY_DATASET'; % Veri setinin ana klasör yolu

%% Görüntüleri oku
imds = imageDatastore(dataPath, ...          % Görüntü datastore oluştur
    'IncludeSubfolders', true, ...            % Alt klasörleri dahil et
    'LabelSource', 'foldernames');            % Etiketleri klasör isimlerinden al

disp("Class counts:");                        % Sınıf dağılımını yazdır
disp(countEachLabel(imds));                  % Her sınıftaki örnek sayısı

%% Split: 70/15/15
rng(42);                                     % Rastgelelik için sabit seed
[imdsTrain, imdsTemp] = splitEachLabel(imds, 0.70, 'randomized'); % %70 eğitim
[imdsVal,   imdsTest] = splitEachLabel(imdsTemp, 0.50, 'randomized'); % %15 doğrulama, %15 test

disp("Split counts (Train):"); disp(countEachLabel(imdsTrain)); % Train dağılımı
disp("Split counts (Val):");   disp(countEachLabel(imdsVal));   % Validation dağılımı
disp("Split counts (Test):");  disp(countEachLabel(imdsTest));  % Test dağılımı

%% ResNet18
net = resnet18;                              % Önceden eğitilmiş ResNet18 modeli
lgraph = layerGraph(net);                    % Katman grafiğine çevir
inputSize = net.Layers(1).InputSize;         % Giriş boyutu [224 224 3]

numClasses = numel(categories(imdsTrain.Labels)); % Sınıf sayısı

lgraph = replaceLayer(lgraph, 'fc1000', ...   % Son fully connected katmanı değiştir
    fullyConnectedLayer(numClasses, 'Name','fcNew', ...
    'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10));

lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', ... % Sınıflandırma katmanını değiştir
    classificationLayer('Name','classoutput'));

%% Augmentation (sadece train)
augmenter = imageDataAugmenter( ...           % Veri artırma ayarları
    'RandRotation', [-25 25], ...              % Rastgele döndürme
    'RandXReflection', true, ...               % Yatay yansıtma
    'RandXTranslation', [-20 20], ...          % X ekseni kaydırma
    'RandYTranslation', [-20 20], ...          % Y ekseni kaydırma
    'RandScale', [0.8 1.2]);                   % Ölçekleme

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ... % Eğitim verisi + augmentation
    'DataAugmentation', augmenter);

augVal  = augmentedImageDatastore(inputSize, imdsVal);  % Validation verisi
augTest = augmentedImageDatastore(inputSize, imdsTest); % Test verisi

%% Training options (stabil)
miniBatch = 16;                               % Mini-batch boyutu
valFreq = max(1, floor(numel(imdsTrain.Files) / miniBatch)); % Validation sıklığı

options = trainingOptions('sgdm', ...         % SGD with momentum
    'MiniBatchSize', miniBatch, ...            % Batch boyutu
    'MaxEpochs', 25, ...                       % Epoch sayısı
    'InitialLearnRate', 5e-5, ...              % Başlangıç öğrenme oranı
    'LearnRateSchedule','piecewise', ...       % Parçalı öğrenme oranı
    'LearnRateDropFactor', 0.1, ...            % Düşürme katsayısı
    'LearnRateDropPeriod', 7, ...              % Kaç epochta bir düşecek
    'Shuffle','every-epoch', ...               % Her epoch karıştır
    'ValidationData', augVal, ...              % Doğrulama verisi
    'ValidationFrequency', valFreq, ...        % Doğrulama sıklığı
    'Verbose', true, ...                       % Detaylı çıktı
    'Plots', 'training-progress', ...          % Eğitim grafiği
    'ExecutionEnvironment','gpu');             % GPU üzerinde eğit

%% Train
trainedNet = trainNetwork(augTrain, lgraph, options); % Modeli eğit

%% Test
[YPred, scores] = classify(trainedNet, augTest); % Test verisini sınıflandır
YTrue = imdsTest.Labels;                        % Gerçek etiketler

accuracy = mean(YPred == YTrue) * 100;          % Doğruluk hesabı
fprintf('\nTest Accuracy: %.2f%%\n', accuracy);

conf = max(scores, [], 2);                      % En yüksek skorlar
fprintf('Average Confidence: %.2f%%\n', mean(conf)*100);

%% Confusion Matrix
figure;                                        % Yeni figür
confusionchart(YTrue, YPred, 'Normalization','row-normalized'); % Normalize confusion matrix
title('Confusion Matrix');                     % Başlık

%% Save
save('C:\Users\LENOVO\Desktop\matlab_final_2\plantLeafDiseaseNet.mat', 'trainedNet'); % Modeli kaydet
disp("Model saved to matlab_final_2");          % Kaydetme mesajı
