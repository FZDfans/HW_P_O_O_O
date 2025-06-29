% 假设点云数据存储在 'pointcloud.txt' 文件中
filename = 'jgg_2.xyz';

% 读取文件内容
data = dlmread(filename, ' ', 0, 0); % 假设数据以空格分隔

% 提取前 3 列（坐标信息）
coordinates = data(:, 1:3);

% 保存为 .mat 文件
save('coordinates.mat', 'coordinates');

disp('坐标信息已保存到 coordinates.mat 文件中');

% 加载数据并验证格式
data = load('coordinates.mat');

% 检查数据是否为结构体，并提取点云矩阵
if isstruct(data)
    field_name = fieldnames(data);
    cleanedPts = data.(field_name{1});
else
    cleanedPts = data;
end

% 验证数据格式
if ~ismatrix(cleanedPts) || size(cleanedPts, 2) ~= 3
    error('输入数据必须是N×3的矩阵');
end

% RANSAC 平面拟合
[planePoints, outlierPoints] = ransacPlaneFitting(cleanedPts);

% 将筛选后的内点云保存为 .mat 文件
save('ransaced_ed_ed.mat', 'planePoints');
disp('筛选后的内点云已保存到 ransaced_ed_ed.mat 文件中');

% 可视化原始点云和筛选后的内点云
visualizeRansacResults(cleanedPts, planePoints, outlierPoints);

% RANSAC 平面拟合函数
function [inlierPoints, outlierPoints] = ransacPlaneFitting(points)
    % 输入验证
    if ~isa(points, 'double') && ~isa(points, 'single')
        points = double(points); % 强制转换为浮点数
    end
    
    % 创建点云对象
    ptCloud = pointCloud(points);
    
    % RANSAC 平面拟合参数
    maxDistance = 10; % 平面距离阈值
    
    % 平面拟合
    [~, inlierIndices, outlierIndices] = pcfitplane(ptCloud, maxDistance,[0,0,1]);
    
    % 提取内点和外点
    inlierPoints = points(inlierIndices, :);
    outlierPoints = points(outlierIndices, :);
end

% 可视化 RANSAC 前后结果
function visualizeRansacResults(originalPoints, inlierPoints, outlierPoints)
    % 可视化原始点云
    figure('Name', '原始点云');
    scatter3(originalPoints(:, 1), originalPoints(:, 2), originalPoints(:, 3), 10, [0.7 0.7 0.7], 'filled');
    axis equal;
    grid on;
    view(3);
    title('原始点云');
    
    % 可视化 RANSAC 筛选后的内点云
    figure('Name', 'RANSAC 筛选后的内点云');
    scatter3(inlierPoints(:, 1), inlierPoints(:, 2), inlierPoints(:, 3), 10, 'b', 'filled');
    axis equal;
    grid on;
    view(3);
    title('RANSAC 筛选后的内点云');
    
    % 可视化 RANSAC 筛选后的外点云
    figure('Name', 'RANSAC 筛选后的外点云');
    scatter3(outlierPoints(:, 1), outlierPoints(:, 2), outlierPoints(:, 3), 10, 'r', 'filled');
    axis equal;
    grid on;
    view(3);
    title('RANSAC 筛选后的外点云');
end
% 1. 加载数据并验证格式
data = load('ransaced_ed_ed.mat');

% 检查数据是否为结构体，并提取点云矩阵
if isstruct(data)
    field_name = fieldnames(data);
    cleanedPts = data.(field_name{1});
else
    cleanedPts = data;
end

% 验证数据格式
if ~ismatrix(cleanedPts) || size(cleanedPts, 2) ~= 3
    error('输入数据必须是N×3的矩阵');
end

% 2. 提取平面法向量
[planeNormal, planePoints] = functak(cleanedPts);

% 3. 改进的孔洞检测方法（现在只返回最大孔洞）
[holeBoundary, holePoints, binaryImg, filledImg, holeImg, holeBoundaryImg] = improvedHoleDetection(cleanedPts, planeNormal, 1000);

% 4. 显示结果
disp(['平面法向量: [', num2str(planeNormal), ']']);
disp(['检测到的孔洞边界点数: ', num2str(size(holeBoundary, 1))]);
disp(['检测到的孔洞区域点数: ', num2str(size(holePoints, 1))]);

% 可视化结果
visualizeResults(cleanedPts, planePoints, holeBoundary, holePoints, binaryImg, filledImg, holeImg, holeBoundaryImg);

% 平面拟合函数
function [planeNormal, planePoints] = functak(points)
    % 输入验证
    if ~isa(points, 'double') && ~isa(points, 'single')
        points = double(points); % 强制转换为浮点数
    end
    
    % 创建点云对象
    ptCloud = pointCloud(points);
    
    % RANSAC平面拟合参数
    maxDistance = 1;       % 平面距离阈值
    referenceVector = [0, 0, 1]; % 参考向量
    
    % 平面拟合
    [planeModel, inlierIndices] = pcfitplane(ptCloud, maxDistance, referenceVector);
    
    % 提取结果
    planeNormal = planeModel.Normal;
    planePoints = points(inlierIndices, :);
end

% 改进的孔洞检测方法（现在只保留最大孔洞）
function [holeBoundary, holePoints, binaryImg, filledImg, holeImg, holeBoundaryImg] = improvedHoleDetection(points, planeNormal, areaThreshold)
    % 输入验证
    if size(points, 2) ~= 3 || size(planeNormal, 2) ~= 3
        error('输入数据维度不正确');
    end

    % 拟合平面公式 Ax + By + Cz + D = 0
    k = -mean(points * planeNormal'); % 自动计算 k
    distances = abs(points * planeNormal' + k);

    % 仅保留接近平面的点
    distanceThreshold = 1000; % 增加距离阈值
    closePoints = points(distances < distanceThreshold, :);

    % 将点投影到主平面
    projectedPoints = closePoints - (closePoints * planeNormal' + k) .* planeNormal;

    % 创建二维网格图像
    gridResolution = 0.3; % 网格分辨率
    xMin = min(projectedPoints(:, 1)); xMax = max(projectedPoints(:, 1));
    yMin = min(projectedPoints(:, 2)); yMax = max(projectedPoints(:, 2));
    xEdges = xMin:gridResolution:xMax + gridResolution / 2;
    yEdges = yMin:gridResolution:yMax + gridResolution / 2;

    % 二维图像投影
    [counts, ~, ~] = histcounts2(projectedPoints(:, 1), projectedPoints(:, 2), xEdges, yEdges);
    binaryImg = counts > 0;

    % 显示二维化图像
    figure('Name', '二维化图像');
    imagesc(binaryImg);
    colormap gray;
    title('二维化图像');
    axis equal;

    % 闭运算
    binaryImgClosed = imclose(binaryImg, strel('disk', 3));
    figure('Name', '闭运算结果');
    imagesc(binaryImgClosed);
    colormap gray;
    title('闭运算结果');
    axis equal;

    % 开运算
    binaryImgOpened = imopen(binaryImgClosed, strel('disk', 1));
    figure('Name', '开运算结果');
    imagesc(binaryImgOpened);
    colormap gray;
    title('开运算结果');
    axis equal;

    % 填充孔洞
    filledImg = imfill(binaryImgOpened, 'holes');
    figure('Name', '填充孔洞后的图像');
    imagesc(filledImg);
    colormap gray;
    title('填充孔洞后的图像');
    axis equal;

    % 检测孔洞
    holeImg = filledImg & ~binaryImgOpened;
    
    % 找到所有连通区域
    cc = bwconncomp(holeImg);
    
    % 计算每个孔洞的面积
    stats = regionprops(cc, 'Area');
    areas = [stats.Area];
    
    % 如果没有孔洞，返回空结果
    if isempty(areas)
        holeBoundary = [];
        holePoints = [];
        holeBoundaryImg = false(size(holeImg));
        return;
    end
    
    % 找到最大孔洞
    [~, maxIdx] = max(areas);
    
    % 创建只包含最大孔洞的图像
    largestHoleImg = false(size(holeImg));
    largestHoleImg(cc.PixelIdxList{maxIdx}) = true;
    holeImg = largestHoleImg;
    
    figure('Name', '最大孔洞图像');
    imagesc(holeImg);
    colormap gray;
    title('最大孔洞图像');
    axis equal;

    % 孔洞边界
    holeBoundaryImg = bwperim(holeImg);
    figure('Name', '最大孔洞边界图像');
    imagesc(holeBoundaryImg);
    colormap gray;
    title('最大孔洞边界图像');
    axis equal;

    % 构建网格坐标（注意顺序）
    [Y, X] = meshgrid(yEdges(1:end-1) + gridResolution / 2, xEdges(1:end-1) + gridResolution / 2);

    % 提取最大孔洞边界点
    boundaryIdx = find(holeBoundaryImg);
    [row, col] = ind2sub(size(holeBoundaryImg), boundaryIdx);
    boundaryX = X(sub2ind(size(X), row, col));
    boundaryY = Y(sub2ind(size(Y), row, col));
    boundaryZ = -(planeNormal(1) * boundaryX + planeNormal(2) * boundaryY + k) / planeNormal(3);
    holeBoundary = [boundaryX(:), boundaryY(:), boundaryZ(:)];

    % 提取最大孔洞区域点
    holeIdx = find(holeImg);
    [row, col] = ind2sub(size(holeImg), holeIdx);
    holeX = X(sub2ind(size(X), row, col));
    holeY = Y(sub2ind(size(Y), row, col));
    holeZ = -(planeNormal(1) * holeX + planeNormal(2) * holeY + k) / planeNormal(3);
    holePoints = [holeX(:), holeY(:), holeZ(:)];  % 修正这里，使用holeY而不是boundaryY
end
% 可视化结果
function visualizeResults(originalPoints, planePoints, holeBoundary, holePoints, binaryImg, filledImg, holeImg, holeBoundaryImg)
    % 可视化原始点云
    figure('Name', '原始点云');
    scatter3(originalPoints(:, 1), originalPoints(:, 2), originalPoints(:, 3), 10, [0.7 0.7 0.7], 'filled');
    title('原始点云');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    grid on;
    view(3);

    % 可视化拟合平面及内点
    figure('Name', '拟合平面及内点');
    scatter3(planePoints(:, 1), planePoints(:, 2), planePoints(:, 3), 10, 'b', 'filled');
    title('拟合平面及内点');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    grid on;
    view(3);

    % 可视化最大孔洞边界
    figure('Name', '最大孔洞边界');
    scatter3(planePoints(:, 1), planePoints(:, 2), planePoints(:, 3), 10, 'b', 'filled');
    hold on;
    scatter3(holeBoundary(:, 1), holeBoundary(:, 2), holeBoundary(:, 3), 30, 'r', 'filled');
    hold off;
    title('最大孔洞边界');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    grid on;
    view(3);

    % 可视化最大孔洞区域
    figure('Name', '最大孔洞区域');
    scatter3(planePoints(:, 1), planePoints(:, 2), planePoints(:, 3), 10, 'b', 'filled');
    hold on;
    scatter3(holePoints(:, 1), holePoints(:, 2), holePoints(:, 3), 50, 'g', 'filled', 'MarkerFaceAlpha', 0.3);
    hold off;
    title('最大孔洞区域');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    axis equal;
    grid on;
    view(3);
end