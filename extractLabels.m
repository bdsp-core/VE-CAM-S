clear all; clc; format compact; 

T = readtable('captions.csv');  

features = {'Absent sleep features',
            'BiPD',
            'Burst suppression',
            'Diffuse delta',
            'Diffuse theta',
            'Extreme delta brush',
            'Extreme low voltage',
            'GPD',
            'GRDA',
            'Intermittent brief attenuation',
            'LPD',
            'LRDA'
            'Moderately low voltage',
            'NCSE',
            'Unreactive'};


% Preallocate a matrix with zeros
featureMatrix = zeros(height(T), numel(features));

% Loop over each caption and check for each feature
for i = 1:height(T)
    str = T.cap(i); 
    str = str{1};

    for j = 1:numel(features)
        if contains(str, features{j})
            featureMatrix(i, j) = 1;
        end
    end
end

% Create a table with featureMatrix and features as column names
featureTable = array2table(featureMatrix, 'VariableNames', features);

% Add the filenames as the first column of the new table
featureTable = addvars(featureTable, T.fn, 'Before', 1);

% Display the feature table
disp(featureTable);
writetable(featureTable,'FeaturesInAppendixFigures.csv')
