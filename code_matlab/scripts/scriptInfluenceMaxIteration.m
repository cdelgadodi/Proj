clear all;
close all;
clc;
addpath('../');


%-- parameters
maxIter = 20;   %-- maximum number of iterations


%-- mnist database location
url = 'https://www.creatis.insa-lyon.fr/~bernard/ge/';
local_data_path = '../data/';
local_param_path = '../param/';


%-- Downlad minst database
filename_db = 'mnist.mat';
if (~exist([local_data_path,filename_db],'file'))
     tools.download(filename_db,url,local_data_path);
end


%-- Load mnist database
load([local_data_path,filename_db]);
widthDigit = size(training.images,2);
heightDigit = size(training.images,1);


%-- Perform training
num_labels = 10;          %-- 10 labels, from 0 to 9


%-- Create X matrix
X = zeros(size(training.images,3),widthDigit*heightDigit+1);
for k=1:size(training.images,3)
    digit = training.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end


%-- Create y vector
y = training.labels;
[m,n] = size(X);


%-- Load pre-learned parameters
filename_param = 'param_ex1_2.mat';
load([local_param_path,filename_param]);

%-- Set Initial theta
initial_theta = zeros(1,n);
 

%Evolution de la mesure de précision calculée à partir de la base de
%données de test

x = 20:18:200

all_theta = zeros(num_labels,n);
%-- Set Initial theta
initial_theta = zeros(1,n);
acc = zeros(1,11);
i = 1;
for maxIter = 20:18:200
    %-- Run gradient descent method to update theta values
        options = struct('MaxIter',maxIter,'epsilon',0.01,'tau',1);
    for c=1:num_labels
        
        [theta] = lrc.gradient_descent(@(t)(lrc.lrCostFunction(t, X, (y == c-1))), initial_theta, options);
        
        %-- Store corresponding result into all_theta matrix
        all_theta(c,:) = theta;
        
        [X_des,~,~] = lrc.gradient_descent(@(t)(lrc.lrCostFunction(t, X, (y == c-1))), initial_theta, options);
        %-- Evaluate the performance of the learned method from the full testing database
        %-- Predict for One-Vs-All
    
    pred = lrc.predict(all_theta, X_des);
    acc(:,i)= num2str(mean(double(pred == y)) * 100);
    i = i+1;
    end
end