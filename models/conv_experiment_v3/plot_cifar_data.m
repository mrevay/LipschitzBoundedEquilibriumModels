
set(0,'DefaultLineMarkerSize',28);
% set(0,'DefaultLineWidth',4);
%%

% Compare mon and lben for different bounds
i50 = load('./identity_conv_w81_L50.0.mat')
i8 = load('./identity_conv_w81_L8.0.mat')
i5 = load('./identity_conv_w81_L5.0.mat')
i3 = load('./identity_conv_w81_L3.0.mat')
i2 = load('./identity_conv_w81_L2.0.mat')
i1 = load('./identity_conv_w81_L1.0.mat')
i0 = load('./identity_conv_w81.mat')

f50 = load('./full_conv_w81_L50.0.mat')
f5 = load('./full_conv_w81_L5.0.mat')
f3 = load('./full_conv_w81_L3.0.mat')
f2 = load('./full_conv_w81_L2.0.mat')
f1 = load('./full_conv_w81_L1.0.mat')
f0 = load('./full_conv_w81.mat')

%%
figure 
hold on
plot(f0.Lipschitz, f0.nominal, 'x', 'LineWidth', 4);
plot(f50.Lipschitz, f50.nominal, 'x', 'LineWidth', 4);
plot(f5.Lipschitz, f5.nominal, 'x', 'LineWidth', 4);
plot(f3.Lipschitz, f3.nominal, 'x', 'LineWidth', 4);
plot(f2.Lipschitz, f2.nominal, 'x', 'LineWidth', 4);
plot(f1.Lipschitz, f1.nominal, 'x', 'LineWidth', 4);

plot(i0.Lipschitz, i0.nominal,'o', 'LineWidth', 4);
plot(i50.Lipschitz, i50.nominal, 'o', 'LineWidth', 4);
% plot(i8.Lipschitz, i8.nominal, 'o', 'LineWidth', 4);
plot(i5.Lipschitz, i5.nominal, 'o', 'LineWidth', 4);
plot(i3.Lipschitz, i3.nominal, 'o', 'LineWidth', 4);
plot(i2.Lipschitz, i2.nominal, 'o', 'LineWidth', 4);
plot(i1.Lipschitz, i1.nominal, 'o', 'LineWidth', 4);

ax = gca
ax.XScale = 'log'
ax.YScale = 'log'

xlabel('Lipschitz Constant')
ylabel('Test Performance')

grid on 
box on

%% Plot Training and Validaiton Curves for f0 and i0
% figure 
% hold on
% plot(f5.train, 'LineWidth', 1.5)
% plot(i5.train, 'LineWidth', 1.5)

figure 
hold on
% plot(f0.val, 'LineWidth', 1.5)
% plot(i0.val, 'LineWidth', 1.5)


plot(f1.val, 'LineWidth', 1.5)
plot(i1.val, 'LineWidth', 1.5)

plot(f2.val, 'LineWidth', 1.5)
plot(i2.val, 'LineWidth', 1.5)

plot(f3.val, 'LineWidth', 1.5)
plot(i3.val, 'LineWidth', 1.5)


plot(f5.val, 'LineWidth', 1.5)
plot(i5.val, 'LineWidth', 1.5)

plot(f50.val, 'LineWidth', 1.5)
plot(i50.val, 'LineWidth', 1.5)

% plot(f0.val, 'LineWidth', 1.5)
% plot(i0.val, 'LineWidth', 1.5)

%% Robustness curves
ff_conv = load('ff_conv_w81.mat');
f0 = load('full_conv_w81.mat');
f1 = load('full_conv_w81_L1.0.mat');
f5 = load('full_conv_w81_L5.0.mat');

figure
hold on
plot(ff_conv.errors)
plot(f0.errors)
plot(f1.errors)
plot(f5.errors)
