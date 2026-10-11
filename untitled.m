%% plot_results.m
% Plots every result produced by the ZTM project.
% Run from the project root (the folder that contains "outputs"):
%     >> cd C:\Users\Anindita\Downloads\ztm_project\ztm_project
%     >> plot_results
% Needs: MATLAB R2016b or newer (no extra toolboxes).
% Reads : outputs/bench_forecast.csv, bench_forecast_folds.csv,
%         bench_anomaly.csv, task3_rl_metrics.json
% Writes: outputs/figures/*.png

clear; clc; close all;
outDir = 'C:\Users\Anindita\Downloads\ztm_project\ztm_project\outputs';
figDir = fullfile(outDir, 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end


%% ---------- 1. FORECASTING: RMSE with error bars ----------
F = readtable(fullfile(outDir, 'bench_forecast.csv'));
F = sortrows(F, 'rmse_mean');
n = height(F); x = 1:n;

figure('Name', 'Forecast RMSE', 'Color', 'w');
bar(x, F.rmse_mean, 0.6); hold on;
errorbar(x, F.rmse_mean, F.rmse_std, 'k.', 'LineWidth', 1.3);
set(gca, 'XTick', x, 'XTickLabel', F.model); xtickangle(30);
ylabel('RMSE (kbps)  -  lower is better');
title('Task 1: next-step throughput forecast (5-fold, grouped by session)');
grid on; box on;
saveas(gcf, fullfile(figDir, 'forecast_rmse.png'));

%% ---------- 2. FORECASTING: R^2 ----------
figure('Name', 'Forecast R2', 'Color', 'w');
bar(x, F.r2_mean, 0.6); hold on;
errorbar(x, F.r2_mean, F.r2_std, 'k.', 'LineWidth', 1.3);
set(gca, 'XTick', x, 'XTickLabel', F.model); xtickangle(30);
ylabel('R^2  -  higher is better'); ylim([0 1]);
title('Task 1: variance explained (R^2)');
grid on; box on;
saveas(gcf, fullfile(figDir, 'forecast_r2.png'));

%% ---------- 3. FORECASTING: per-fold RMSE (shows models move together) ----------
T = readtable(fullfile(outDir, 'bench_forecast_folds.csv'));
U = unstack(T(:, {'fold', 'model', 'rmse'}), 'rmse', 'model');
models = U.Properties.VariableNames(2:end);
figure('Name', 'Per-fold RMSE', 'Color', 'w'); hold on;
for k = 1:numel(models)
    plot(U.fold + 1, U.(models{k}), '-o', 'LineWidth', 1.6, 'DisplayName', models{k});
end
xlabel('Cross-validation fold'); ylabel('RMSE (kbps)');
title('Per-fold RMSE: models rise and fall together (fold difficulty dominates)');
legend('Location', 'best'); grid on; box on;
xticks(unique(U.fold + 1));
saveas(gcf, fullfile(figDir, 'forecast_per_fold.png'));

%% ---------- 4. TRAINING TIME (median, robust to one slow fold) ----------
G = groupsummary(T, 'model', 'median', 'train_s');
G = sortrows(G, 'median_train_s');
figure('Name', 'Training time', 'Color', 'w');
barh(G.median_train_s); set(gca, 'YTick', 1:height(G), 'YTickLabel', G.model, 'XScale', 'log');
xlabel('Median training time per fold (seconds, log scale)');
title('Cost of each forecasting model');
grid on; box on;
saveas(gcf, fullfile(figDir, 'forecast_train_time.png'));

%% ---------- 5. ANOMALY: precision / recall / F1 ----------
A = readtable(fullfile(outDir, 'bench_anomaly.csv'));
A = sortrows(A, 'f1_mean', 'descend');
na = height(A); xa = 1:na;

figure('Name', 'Anomaly P/R/F1', 'Color', 'w');
bar(xa, [A.precision_mean A.recall_mean A.f1_mean], 'grouped');
set(gca, 'XTick', xa, 'XTickLabel', A.model); xtickangle(30);
ylabel('Score'); ylim([0 1]);
legend({'Precision', 'Recall', 'F1'}, 'Location', 'northeast');
title('Task 2: anomaly (throughput collapse) detection');
grid on; box on;
saveas(gcf, fullfile(figDir, 'anomaly_prf1.png'));

%% ---------- 6. ANOMALY: ROC-AUC and PR-AUC ----------
figure('Name', 'Anomaly AUC', 'Color', 'w');
bar(xa, [A.roc_auc_mean A.pr_auc_mean], 'grouped');
set(gca, 'XTick', xa, 'XTickLabel', A.model); xtickangle(30);
ylabel('AUC'); ylim([0 1]);
legend({'ROC-AUC', 'PR-AUC'}, 'Location', 'northeast');
title('Task 2: ranking quality (threshold-free)');
grid on; box on;
saveas(gcf, fullfile(figDir, 'anomaly_auc.png'));

%% ---------- 7. RESOURCE ALLOCATION (Q-learning) ----------
R = jsondecode(fileread(fullfile(outDir, 'task3_rl_metrics.json')));
pol = fieldnames(R.policies);
meanR = zeros(numel(pol), 1); stdR = zeros(numel(pol), 1);
for k = 1:numel(pol)
    meanR(k) = R.policies.(pol{k}).mean_reward;
    stdR(k)  = R.policies.(pol{k}).std_reward;
end
[meanR, order] = sort(meanR, 'descend'); stdR = stdR(order); pol = pol(order);

figure('Name', 'RL policies', 'Color', 'w');
bar(1:numel(pol), meanR, 0.6); hold on;
errorbar(1:numel(pol), meanR, stdR, 'k.', 'LineWidth', 1.3);
set(gca, 'XTick', 1:numel(pol), 'XTickLabel', strrep(pol, '_', ' ')); xtickangle(30);
ylabel('Mean reward (closer to 0 is better)');
title('Task 3: allocation policies on held-out episodes');
grid on; box on;
saveas(gcf, fullfile(figDir, 'rl_policies.png'));

figure('Name', 'RL learning curve', 'Color', 'w');
plot(R.reward_curve, '-o', 'LineWidth', 1.8);
xlabel('Training epoch'); ylabel('Average reward');
title('Task 3: Q-learning training curve');
grid on; box on;
saveas(gcf, fullfile(figDir, 'rl_learning_curve.png'));

fprintf('Done. %d figures saved in %s\n', 7, figDir);