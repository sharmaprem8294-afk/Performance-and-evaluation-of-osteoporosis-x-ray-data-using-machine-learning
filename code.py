clc;
clear;
close all;

1. Load Augmented Dataset
dataPath = 'C:\Users\kruti\OneDrive\Desktop\KAPISH\SEM 7\Honours\Augmented_Dataset';
imds = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

disp('Dataset loaded successfully');
disp(countEachLabel(imds));

2. Train-Test Split 
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

3. Feature Extraction Using CNN 
net = mobilenetv2;
inputSize = net.Layers(1).InputSize(1:2);
featureLayer = 'global_average_pooling2d_1';

augTrain = augmentedImageDatastore(inputSize, imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');
augTest  = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

Xtrain = activations(net, augTrain, featureLayer, 'OutputAs', 'rows');
Xtest  = activations(net, augTest, featureLayer, 'OutputAs', 'rows');

Ytrain = imdsTrain.Labels;
Ytest  = imdsTest.Labels;

disp('Feature extraction completed.');

4. Save Numeric Dataset 
featureNames = compose("Feature_%d", 1:size(Xtrain,2));

trainTable = array2table(Xtrain, 'VariableNames', featureNames);
trainTable.Label = Ytrain;

testTable = array2table(Xtest, 'VariableNames', featureNames);
testTable.Label = Ytest;

writetable(trainTable, 'numeric_train_data.csv');
writetable(testTable, 'numeric_test_data.csv');

disp('Numeric datasets saved as CSV files.');

5. Train Classification Models
models = cell(1,5);
modelNames = {'Logistic Regression','Decision Tree','SVM','AdaBoost','Random Forest'};

% Logistic Regression
models{1} = fitclinear(Xtrain, Ytrain, ...
    'Learner','logistic','Solver','lbfgs');

% Decision Tree
models{2} = fitctree(Xtrain, Ytrain, 'MinLeafSize', 5);

% Support Vector Machine
models{3} = fitcsvm(Xtrain, Ytrain, ...
    'KernelFunction','rbf','KernelScale','auto');

% AdaBoost
treeTemplate = templateTree('MaxNumSplits',20);
models{4} = fitcensemble(Xtrain, Ytrain, ...
    'Method','AdaBoostM1','Learners',treeTemplate,'NumLearningCycles',100);

% Random Forest
rfTemplate = templateTree('MinLeafSize',5);
models{5} = fitcensemble(Xtrain, Ytrain, ...
    'Method','Bag','Learners',rfTemplate,'NumLearningCycles',100);

disp('All models trained successfully.');

6. Model Accuracy Evaluation 
accuracy = zeros(1,5);
Ypred = cell(1,5);

fprintf('\nModel Accuracy:\n');
for i = 1:5
    Ypred{i} = predict(models{i}, Xtest);
    accuracy(i) = mean(Ypred{i} == Ytest) * 100;
    fprintf('%s: %.2f%%\n', modelNames{i}, accuracy(i));
end

figure;
bar(accuracy);
set(gca,'XTickLabel',modelNames,'XTickLabelRotation',45);
ylabel('Accuracy (%)');
title('Classifier Accuracy Comparison');
grid on;

7. Confusion Matrices 
figure;
for i = 1:5
    subplot(2,3,i);
    cm = confusionmat(Ytest, Ypred{i});
    heatmap(categories(Ytest), categories(Ytest), cm);
    title(['Confusion Matrix - ' modelNames{i}]);
end

8. Precision, Recall, F1 Score 
precision = zeros(1,5);
recall    = zeros(1,5);
f1Score   = zeros(1,5);

for i = 1:5
    cm = confusionmat(Ytest, Ypred{i});
    tp = cm(2,2); fp = cm(1,2);
    fn = cm(2,1); tn = cm(1,1);

    precision(i) = tp / (tp + fp + eps);
    recall(i)    = tp / (tp + fn + eps);
    f1Score(i)   = 2 * (precision(i)*recall(i)) / (precision(i)+recall(i)+eps);
end

figure;
bar([precision; recall; f1Score]');
set(gca,'XTickLabel',modelNames,'XTickLabelRotation',45);
legend('Precision','Recall','F1 Score');
ylabel('Score');
title('Performance Metrics');
grid on;

9. ROC Curve & AUC 
figure; hold on;
colors = {'r','g','b','m','k'};

for i = 1:5
    [~, score] = predict(models{i}, Xtest);
    [Xroc,Yroc,~,AUC] = perfcurve(Ytest, score(:,2), categories(Ytest){2});
    plot(Xroc, Yroc, colors{i}, 'LineWidth',2, ...
        'DisplayName', sprintf('%s (AUC=%.3f)',modelNames{i},AUC));
end

xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('ROC Curves Comparison');
legend('Location','southeast');
grid on;
hold off;

disp('All evaluations completed successfully.');
