function plotDecisionBoundary(theta, X, y)
%PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
%   positive examples and o for the negative examples. X is assumed to be 
%   a either 
%   1) Mx3 matrix, where the first column is an all-ones column for the 
%      intercept.
%   2) MxN, N>3 matrix, where the first column is all-ones

% Plot Data
plotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculate the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plot, and adjust axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary')
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    % Notice you need to specify the range [0, 0]
    % countour 通常用来画等高线。u、v是步长，z是等高线的值。
    % 后面的[0, 0]表示只花值为0的等高线，也就是decision boundary。
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

%%  Self-written contour test code:
% a = linspace(0, 1, 11);
% b = linspace(0, 1, 11);
% c = zeros(length(a), length(b));
% % 如果没有下面这部分，直接用0矩阵执行contour函数，会提示警告并且没有等高线。
% c = c';
% for i = 1:length(a)
%     for j = 1:length(b)
%         c(i,j) = a(i) + b(j);
%     end
% end
% % c 不能为常量，需要通过其他两个变量赋值，否则会提示：ZData 为常量时，无法显示等高线
% contour(a,b,c);

end


