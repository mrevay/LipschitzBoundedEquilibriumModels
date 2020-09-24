clc
clear

g02 = load('fc_lip0.2_w80.mat')
g03 = load('fc_lip0.4_w80.mat')
g03 = load('fc_lip0.3_w80.mat')
g04 = load('fc_lip0.4_w80.mat')
g05 = load('fc_lip0.5_w80.mat')
g08 = load('fc_lip0.8_w80.mat')
g5 = load('fc_lip5.0_w80.mat')
g8 = load('fc_lip8.0_w80.mat')
g10 = load('fc_lip10.0_w80.mat')
g50 = load('fc_lip50.0_w80.mat')
mon = load('mon_w80.mat')
ode = load('ode_w80.mat')
uncon = load('uncon_w80.mat')

%%

fig_pos = [-0 -0 1200 900];

fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';

p1 = plot(g02.epsilon, g02.errors, 'LineWidth', 2)
hold on
p2 = plot(g03.epsilon, g03.errors, 'LineWidth', 2)
p3 = plot(g04.epsilon, g04.errors, 'LineWidth', 2)
p4 = plot(g05.epsilon, g05.errors, 'LineWidth', 2)
p5 = plot(g08.epsilon, g08.errors, 'LineWidth', 2)
p6 = plot(g5.epsilon, g5.errors, 'LineWidth', 2)


p7 = plot(mon.epsilon, mon.errors, 'k--', 'LineWidth', 2)
p8 = plot(uncon.epsilon, uncon.errors, 'k-.', 'LineWidth', 2)
p9 = plot(ode.epsilon, ode.errors, 'k:', 'LineWidth', 2)

% p10 = plot(g50.epsilon, g50.errors, 'LineWidth', 2)
% p10 = plot(g8.epsilon, g8.errors, 'LineWidth', 2)
% p10 = plot(g10.epsilon, g10.errors, 'LineWidth', 2)

grid on
box on
legend({'$\gamma=0.2$', '$\gamma=0.3$', '$\gamma=0.4$', '$\gamma=0.5$',...
        '$\gamma=0.8$', '$\gamma=5.0$', 'monotone', 'unconstrained', 'ode', '$\gamma=50$'},...
        'Interpreter', 'Latex', 'Location', 'Northwest')
    
xlabel('$\ell_2$ perturbation', 'Interpreter', 'Latex')
ylabel('Error Rate', 'Interpreter', 'Latex')

%%
print(fig, '-dpdf', 'mnist_robustness', '-bestfit');