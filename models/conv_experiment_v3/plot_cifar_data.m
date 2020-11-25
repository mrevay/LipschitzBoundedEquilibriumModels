
clc
clear

set(0,'DefaultLineMarkerSize',28);
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(0,'defaultTextInterpreter','latex');
set(0,'defaultAxesFontSize',28)
set(0, 'DefaultLineLineWidth', 1.5);

c1 = [0, 0.4470, 0.7410];
c2 = [0.8500, 0.3250, 0.0980];
c3 = [0.9290, 0.6940, 0.1250];
c4 = [0.4940, 0.1840, 0.5560];
c5 = [0.4660, 0.6740, 0.1880];
c6 = [0.3010, 0.7450, 0.9330];
c7 = [0.6350,    0.0780,    0.1840];

%%

ff = load('./ff_conv_w81.mat')

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
fig_pos = [232 266 942 780];

fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on

ms = 3

p0 = plot(f0.Lipschitz, f0.nominal, '*', 'LineWidth', ms, 'color', c1);
p1 = plot(f50.Lipschitz, f50.nominal, 'x', 'LineWidth', ms, 'color', c1);
plot(f5.Lipschitz, f5.nominal, 'x', 'LineWidth', ms, 'color', c1);
plot(f3.Lipschitz, f3.nominal, 'x', 'LineWidth', ms, 'color', c1);
plot(f2.Lipschitz, f2.nominal, 'x', 'LineWidth', ms, 'color', c1);
plot(f1.Lipschitz, f1.nominal, 'x', 'LineWidth', ms, 'color', c1);


p2 = plot(i50.Lipschitz, i50.nominal, 'o', 'LineWidth', ms, 'color', c2);
plot(i5.Lipschitz, i5.nominal, 'o', 'LineWidth', ms, 'color', c2);
plot(i3.Lipschitz, i3.nominal, 'o', 'LineWidth', ms, 'color', c2);
plot(i2.Lipschitz, i2.nominal, 'o', 'LineWidth', ms, 'color', c2);
plot(i1.Lipschitz, i1.nominal, 'o', 'LineWidth', ms, 'color', c2);

p3 = plot(i0.Lipschitz, i0.nominal,'^', 'LineWidth', ms, 'color', c5);

p4 = plot(ff.Lipschitz, ff.nominal, 's', 'LineWidth', ms, 'color', c4);

ax = gca
ax.XScale = 'log'
ax.YScale = 'log'

legend([p0, p1, p2, p3, p4], {"LBEN $_{\gamma<\infty}$", "LBEN", 'LBEN $_{\Lambda=I}$', 'MON', 'Feedforward'})

xlabel('Lipschitz (lower bound)', 'Interpreter', 'Latex')
ylabel('Test Error', 'Interpreter', 'Latex')

print(fig, '-dpdf', 'mon_vs_lben', '-bestfit');


%% Plot showing respresentational power for the two models
xdata1 = [f1.Lipschitz, f2.Lipschitz, f3.Lipschitz, f5.Lipschitz, f50.Lipschitz, f0.Lipschitz];
xdata2 = [i1.Lipschitz, i2.Lipschitz, i3.Lipschitz, i5.Lipschitz, i50.Lipschitz, i0.Lipschitz];

ydata1 = [f1.nominal, f2.nominal, f3.nominal, f5.nominal, f50.nominal, f0.nominal];
ydata2 = [i1.nominal, i2.nominal, i3.nominal, i5.nominal, i50.nominal, i0.nominal];


fig_pos = [232 346 840 700];

fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on
plot(xdata1, ydata1,  'LineWidth', 2.0)
plot(xdata2, ydata2,  'LineWidth', 2.0)
legend({'LBEN', 'MON'}, 'Location', 'NorthEast')

ax = gca
ax.XScale = 'log'
ax.YScale = 'log'

xlabel('Lipschitz (lower bound)', 'Interpreter', 'Latex')
ylabel('Test Performance', 'Interpreter', 'Latex')

print(fig, '-dpdf', 'mon_vs_lben', '-bestfit');

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
f1 = load('full_conv_w81_L1.0.mat');
f2 = load('full_conv_w81_L2.0.mat');
f3 = load('full_conv_w81_L3.0.mat');
f5 = load('full_conv_w81_L5.0.mat');
f50 = load('full_conv_w81_L50.0.mat');
ff_conv = load('ff_conv_w81.mat');
f0 = load('full_conv_w81.mat');

fig_pos = [-0 -0 1200 900];
fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on


plot(f1.epsilon,  100*f1.errors, 'LineWidth', 2.0)
plot(f5.epsilon, 100*f5.errors, 'LineWidth', 2.0)
plot(f50.epsilon, 100*f50.errors, 'LineWidth', 2.0)
plot(f0.epsilon, 100*f0.errors, 'LineWidth', 2.0)
plot(ff_conv.epsilon, 100*ff_conv.errors, 'LineWidth', 2.0)

axis([0, 6, 25, 100])
grid on 
box on

legend(["LBEN $\gamma=1.0$", "LBEN $\gamma=5.0$", ...
        "LBEN $\gamma=50.0$", "LBEN $\gamma<\infty$", "Feedforward"], ...
        'Location', "SouthEast", 'Interpreter', 'Latex')
    

xlabel('$\ell_2$ perturbation', 'Interpreter', 'Latex')
ylabel('Test Error (\%)', 'Interpreter', 'Latex')

print(fig, '-dpdf', 'cifar_robustness', '-bestfit');


%% Plot Training Curves
ff = load('./old_with_training_curves/ff_conv_w81.mat')

f0 = load('./old_with_training_curves/full_conv_w81.mat')
f1 = load('./old_with_training_curves/full_conv_w81_L1.0.mat')
f3 = load('./old_with_training_curves/full_conv_w81_L3.0.mat')
f5 = load('./old_with_training_curves/full_conv_w81_L5.0.mat')
f50 = load('./old_with_training_curves/full_conv_w81_L50.0.mat')

i0 = load('./old_with_training_curves/identity_conv_w81.mat')
i1 = load('./old_with_training_curves/identity_conv_w81_L1.0.mat')
i3 = load('./old_with_training_curves/identity_conv_w81_L3.0.mat')
i5 = load('./old_with_training_curves/identity_conv_w81_L5.0.mat')
i50 = load('./old_with_training_curves/identity_conv_w81_L50.0.mat')

fig_pos = [-0 -0 1200 900];
fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on


plot(i1.train, '--','LineWidth', 2.0, 'color', c2)
plot(i5.train, '-.','LineWidth', 2.0, 'color', c2)
plot(i50.train, 'LineWidth', 2.0, 'color', c2)

plot(f1.train, '--', 'LineWidth', 2.0, 'color', c1)
plot(f5.train, '-.', 'LineWidth', 2.0, 'color', c1)
plot(f50.train, 'LineWidth', 2.0, 'color', c1)

plot(f0.train, ':', 'LineWidth', 2.0, 'color', c1)
plot(i0.train, ':', 'LineWidth', 2.0, 'color', c2)

plot(ff.train, 'LineWidth', 2.0, 'color', 'k')


legend(["LBEN $_{\gamma=1.0, ~\Lambda=I}$",...
        "LBEN $_{\gamma=5.0, ~\Lambda=I}$",...
        "LBEN $_{\gamma=50.0, ~\Lambda=I}$",...
        "LBEN $_{\gamma=1.0}$", ...
        "LBEN $_{\gamma=5.0}$", ...
        "LBEN $_{\gamma=50.0}$", ...
        "LBEN $_{\gamma<\infty}$", ...
        "MON",...
        "Feedforward"], ...
        'Location', "NorthEast", ...
        'Interpreter', 'Latex', ...
        'FontSize', 18)
    
xlabel('Epochs', 'Interpreter', 'Latex')
ylabel('Training Error', 'Interpreter', 'Latex')

ax = gca

print(fig, '-dpdf', 'cifar_training', '-bestfit');

%% Plot Training Curves
f0 = load('./old_with_training_curves/full_conv_w81.mat')
f1 = load('./old_with_training_curves/full_conv_w81_L1.0.mat')
f3 = load('./old_with_training_curves/full_conv_w81_L3.0.mat')
f5 = load('./old_with_training_curves/full_conv_w81_L5.0.mat')
f50 = load('./old_with_training_curves/full_conv_w81_L50.0.mat')

i0 = load('./old_with_training_curves/identity_conv_w81.mat')
i1 = load('./old_with_training_curves/identity_conv_w81_L1.0.mat')
i3 = load('./old_with_training_curves/identity_conv_w81_L3.0.mat')
i5 = load('./old_with_training_curves/identity_conv_w81_L5.0.mat')
i50 = load('./old_with_training_curves/identity_conv_w81_L50.0.mat')

fig_pos = [-0 -0 1200 900];
fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';
hold on
grid on
box on


plot(i1.val, '--','LineWidth', 2.0, 'color', c2)
plot(i5.val, '-.','LineWidth', 2.0, 'color', c2)
plot(i50.val, 'LineWidth', 2.0, 'color', c2)

plot(f1.val, '--', 'LineWidth', 2.0, 'color', c1)
plot(f5.val, '-.', 'LineWidth', 2.0, 'color', c1)
plot(f50.val, 'LineWidth', 2.0, 'color', c1)

plot(f0.val, ':', 'LineWidth', 2.0, 'color', c1)
plot(i0.val, ':', 'LineWidth', 2.0, 'color', c2)


legend(["LBEN $_{\gamma=1.0, ~\Lambda=I}$",...
        "LBEN $_{\gamma=5.0, ~\Lambda=I}$",...
        "LBEN $_{\gamma=50.0, ~\Lambda=I}$",...
        "LBEN $_{\gamma=1.0}$", ...
        "LBEN $_{\gamma=5.0}$", ...
        "LBEN $_{\gamma=50.0}$", ...
        "LBEN $_{\gamma=\infty}$", ...
        "MON"], ...
        'Location', "NorthEast", ...
        'Interpreter', 'Latex', ...
        'FontSize', 18)
    
xlabel('Epochs', 'Interpreter', 'Latex')
ylabel('Training Error', 'Interpreter', 'Latex')

% print(fig, '-dpdf', 'cifar_validation', '-bestfit');