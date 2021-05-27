clc
clear

aa0 = load('./models/adversarial_training/ff_w80_eps0.0.mat')
aa0p5 = load('./models/adversarial_training/ff_w80_eps0.5.mat')
aa0p8 = load('./models/adversarial_training/ff_w80_eps0.8.mat')
aa1p0 = load('./models/adversarial_training/ff_w80_eps1.0.mat')
aa1p5 = load('./models/adversarial_training/ff_w80_eps1.5.mat')
aa2p0 = load('./models/adversarial_training/ff_w80_eps2.0.mat')
aa3p0 = load('./models/adversarial_training/ff_w80_eps3.0.mat')
aa5p0 = load('./models/adversarial_training/ff_w80_eps5.0.mat')

%%
res = {}
eps = [0.5, 1, 2, 3];
gamma = [0.2, 1.0, 2.0 ];
for ii = 1:length(eps)
    for jj = 1:length(gamma)
        name = sprintf('./models/adversarial_training/fc_lip%1.1f_w80_eps%1.1f.mat', gamma(jj), eps(ii));
        data = load('./models/adversarial_training/ff_w80_eps0.0.mat');
        res{ii, jj} = data;
    end
end


%%
g02 = load('./models/adversarial_training/fc_lip0.2_w80.mat')
g03 = load('./models/adversarial_training/fc_lip0.3_w80.mat')
g04 = load('./models/adversarial_training/fc_lip0.4_w80.mat')
g05 = load('./models/adversarial_training/fc_lip0.5_w80.mat')
g08 = load('./models/adversarial_training/fc_lip0.8_w80.mat')
g1 = load('./models/adversarial_training/fc_lip1.0_w80.mat')
g2 = load('./models/adversarial_training/fc_lip2.0_w80.mat')
g5 = load('./models/adversarial_training/fc_lip5.0_w80.mat')
g8 = load('./models/adversarial_training/fc_lip8.0_w80.mat')
g10 = load('./models/adversarial_training/fc_lip10.0_w80.mat')
g50 = load('./models/adversarial_training/fc_lip50.0_w80.mat')
mon = load('./models/adversarial_training/mon_w80.mat')
% ode = load('./models/adversarial_training/ode_w80.mat')
uncon = load('./models/adversarial_training/uncon_w80.mat')

%%
range = 1:15

fig_pos = [-0 -0 1200 900];

fig = figure('Position', fig_pos)
fig.PaperPositionMode = 'auto';
fig.PaperOrientation = 'landscape';

p1 = plot(g02.epsilon(range), g02.errors(range), 'LineWidth', 2)
hold on
p2 = plot(g03.epsilon(range), g03.errors(range), 'LineWidth', 2)
p2 = plot(g04.epsilon(range), g04.errors(range), 'LineWidth', 2)
p2 = plot(g05.epsilon(range), g05.errors(range), 'LineWidth', 2)
p3 = plot(g08.epsilon(range), g08.errors(range), 'LineWidth', 2)
p4 = plot(g5.epsilon(range), g5.errors(range), 'LineWidth', 2)


p5 = plot(mon.epsilon(range), mon.errors(range), 'k--', 'LineWidth', 2)
p6 = plot(uncon.epsilon(range), uncon.errors(range), 'k-.', 'LineWidth', 2)
% p7 = plot(ode.epsilon, ode.errors, 'k:', 'LineWidth', 2)

p8 = plot(aa0.epsilon(range), aa0.errors(range), 'LineWidth', 2)
p9 = plot(aa1p0.epsilon(range), aa1p0.errors(range), 'LineWidth', 2)
p10 = plot(aa2p0.epsilon(range), aa2p0.errors(range), 'LineWidth', 2)

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

%% Plot AT models
range = 1:40
plot(aa0.epsilon(range), aa0.errors(range), 'LineWidth', 2)
hold on
plot(aa0p5.epsilon(range), aa0p5.errors(range), 'LineWidth', 2)
plot(aa0p8.epsilon(range), aa0p8.errors(range), 'LineWidth', 2)
plot(aa1p0.epsilon(range), aa1p0.errors(range), 'LineWidth', 2)
plot(aa1p5.epsilon(range), aa1p5.errors(range), 'LineWidth', 2)
plot(aa2p0.epsilon(range), aa2p0.errors(range), 'LineWidth', 2)
plot(aa3p0.epsilon(range), aa3p0.errors(range), 'LineWidth', 2)
plot(aa5p0.epsilon(range), aa5p0.errors(range), 'LineWidth', 2)


%% Plot Lipschitz bounded models

plot(g02.epsilon(range), g02.errors(range), 'LineWidth', 2)
hold on
plot(g03.epsilon(range), g03.errors(range), 'LineWidth', 2)
plot(g04.epsilon(range), g04.errors(range), 'LineWidth', 2)
plot(g05.epsilon(range), g05.errors(range), 'LineWidth', 2)
plot(g08.epsilon(range), g08.errors(range), 'LineWidth', 2)
plot(g1.epsilon(range), g1.errors(range), 'LineWidth', 2)
plot(g5.epsilon(range), g5.errors(range), 'LineWidth', 2)


%% Plot Lipschitz AT bounded models
eps = [0.5, 1, 2, 3];
gamma = [0.2, 1.0, 2.0 ];
figure
hold on
for ii = 1:length(eps)
    for jj = 1:length(gamma)
        plot(res{ii, jj}.epsilon, res{ii, jj}.errors, 'LineWidth', 2);
    end
end


%%
print(fig, '-dpdf', 'mnist_robustness', '-bestfit');