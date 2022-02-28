clc; clear; close all;
format short g

%% Prepare the data
N = 1000;
splitRatio = 0.1;

w_ = rand(3, 1);
X = randi([-N N], N, 2);
Y = sign([X ones(N, 1)]*w_);

trainN = floor(N*splitRatio);
testN = N - trainN;
trainX = X(1:trainN, :);
trainY = Y(1:trainN);
testX = X(trainN+1:end, :);
testY = Y(trainN+1:end);

%% Perceptron Learning Algorithm (PLA)
% Let w=(w1, w2, w3), x=(x1, x2, 1), y={1, -1}
% Assume that f(x; w) = sign(w^T*x)
% Define L(w; x, y) = -sum(y*(w^T*x))
% Given (x, y), our goal is to find the suitable w

w = rand(3, 1);  % initial the parameter
correct = 0;

while correct ~= trainN
    correct = 0;
    for i = 1:trainN
        x = [trainX(i, :), 1]';
        y = trainY(i);

        if sign(w'*x) ~= y
            % update the parameter by GD
            w = w + y*x;
        else
            correct = correct + 1;
        end
    end
end

%% Evaluation
Yhat = sign([testX ones(testN, 1)]*w);
testAcc = sum(Yhat == testY) / testN;
fprintf("test accuracy: %.2f\n", testAcc);

%% Plot the figure
X1 = X(Y>0, :);
X2 = X(Y<0, :);
x2fun =@ (w, x1) -(w(1)*x1 + w(3))/w(2);

figure(1);
% plot training data
plot(X1(:, 1), X1(:, 2), 'o', ...
    X2(:, 1), X2(:, 2), 'x');

% plot the decision boundary
line([N, -N], ...
    [x2fun(w_, N), x2fun(w_, -N)], ...
    'linestyle', '--', 'color', 'k');

% plot the line by PLA
line([N, -N], ...
    [x2fun(w, N), x2fun(w, -N)], ...
    'linestyle', '-', 'color', 'm');

legend('Category1', 'Category2', ...
    'Decision boundary', 'PLA', ...
    'location', 'northeast');

% control the display boundary
xlim([-N, N]);
ylim([-N, N]);
grid on;
