clc
clear

aa0 = load('./models/adversarial_training/ff_w80_eps0.0.mat')
aa0p5 = load('./models/adversarial_training/ff_w80_eps0.5.mat')
aa1p0 = load('./models/adversarial_training/ff_w80_eps1.0.mat')
aa2p0 = load('./models/adversarial_training/ff_w80_eps2.0.mat')
aa5p0 = load('./models/adversarial_training/ff_w80_eps5.0.mat')
aa10p0 = load('./models/adversarial_training/ff_w80_eps10.0.mat')
aa15p0 = load('./models/adversarial_training/ff_w80_eps15.0.mat')
aa20p0 = load('./models/adversarial_training/ff_w80_eps20.0.mat')



%%
g02 = load('./models/adversarial_training/fc_lip0.2_w80.mat')
g03 = load('./models/adversarial_training/fc_lip0.4_w80.mat')
g03 = load('./models/adversarial_training/fc_lip0.3_w80.mat')
g04 = load('./models/adversarial_training/fc_lip0.4_w80.mat')
g05 = load('./models/adversarial_training/fc_lip0.5_w80.mat')
g08 = load('./models/adversarial_training/fc_lip0.8_w80.mat')
g5 = load('./models/adversarial_training/fc_lip5.0_w80.mat')
g8 = load('./models/adversarial_training/fc_lip8.0_w80.mat')
g10 = load('./models/adversarial_training/fc_lip10.0_w80.mat')
g50 = load('./models/adversarial_training/fc_lip50.0_w80.mat')
mon = load('./models/adversarial_training/mon_w80.mat')
% ode = load('./models/adversarial_training/ode_w80.mat')
uncon = load('./models/adversarial_training/uncon_w80.mat')

%%

fig_pos = [-0 -0 1200 900];

fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';

p1 = plot(g02.epsilon, g02.errors, 'LineWidth', 2)
hold on
p2 = plot(g03.epsilon, g03.errors, 'LineWidth', 2)
p2 = plot(g04.epsilon, g04.errors, 'LineWidth', 2)
p2 = plot(g05.epsilon, g05.errors, 'LineWidth', 2)
p3 = plot(g08.epsilon, g08.errors, 'LineWidth', 2)
p4 = plot(g5.epsilon, g5.errors, 'LineWidth', 2)


p5 = plot(mon.epsilon, mon.errors, 'k--', 'LineWidth', 2)
p6 = plot(uncon.epsilon, uncon.errors, 'k-.', 'LineWidth', 2)
% p7 = plot(ode.epsilon, ode.errors, 'k:', 'LineWidth', 2)

p8 = plot(aa0.epsilon, aa0.errors, 'LineWidth', 2)
p9 = plot(aa1p0.epsilon, aa1p0.errors, 'LineWidth', 2)
p10 = plot(aa10p0.epsilon, aa5p0.errors, 'LineWidth', 2)

% p10 = plot(g50.epsilon, g50.errors, 'LineWidth', 2)
% p10 = plot(g8.epsilon, g8.errors, 'LineWidth', 2)
% p10 = plot(g10.epsilon, g10.errors, 'LineWidth', 2)

grid on
box on
legend({'$\gamma=0.2$', '$\gamma=0.3$', ...
        '$\gamma=0.8$', '$\gamma=5.0$', 'monotone', 'unconstrained', 'ode',...
        'aa0', 'aa1.0', 'aa 2.0'},...
        'Interpreter', 'Latex', 'Location', 'Northwest')
    
xlabel('$\ell_2$ perturbation', 'Interpreter', 'Latex')
ylabel('Error Rate', 'Interpreter', 'Latex')

%%
print(fig, '-dpdf', 'mnist_robustness', '-bestfit');