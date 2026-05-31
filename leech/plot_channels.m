% plot_channels.m
% Load a recording, select V2 spikes interactively, plot V1, V2, I2.
% V2 spike times are drawn as red vertical lines in the V2 and I2 panels.
% Run with:  plot_channels

fn = 'C:\Users\user\Documents\MATLAB\LeechCode26\2026_05_29_0024_lastrun_ncell ap.abf';

% --- load data ---
[data, tms, h] = loadephys(fn, 2);

% --- channels (Grn_Vm, Grn_Im, Red_Vm, Red_Im) ---
V1 = data(:,1);
V2 = data(:,3);
I2 = data(:,4);

% --- interactive spike selection on V2 ---
gdsp2 = selspktrace(V2, tms);    % bracket the dots with the green/blue lines, then Done
v2tms = gdsp2.tms;
fprintf('V2 spikes selected: %d\n', numel(v2tms));
if isempty(v2tms)
    warning(['No V2 spikes were selected -> no red lines. ' ...
             'In the pop-up, put one line above the dots and one below, ' ...
             'then click Done.']);
end

% --- crop the x-axis window (seconds; [] = leave that edge at the data limit) ---
t_start = [];              % e.g. 5   to start the plot at 5 s
t_end   = [];              % e.g. 30  to end the plot at 30 s

% --- plot ---
figure('Color', 'w')

ax(1) = subplot(3,1,1);
plot(tms, V1)
ylabel('V1 (mV)')

ax(2) = subplot(3,1,2);
plot(tms, V2)
ylabel('V2 (mV)')

ax(3) = subplot(3,1,3);
plot(tms, I2)
ylabel('I2 (nA)')
xlabel('Time (s)')

linkaxes(ax, 'x');
xl = xlim(ax(1));
if ~isempty(t_start), xl(1) = t_start; end
if ~isempty(t_end),   xl(2) = t_end;   end
xlim(ax(1), xl);

% --- 5% y-padding so peaks aren't clipped + remove box outline ---
for k = 1:numel(ax)
    yl = ylim(ax(k));
    pad = 0.05 * range(yl);
    ylim(ax(k), [yl(1)-pad, yl(2)+pad]);
    box(ax(k), 'off')
end

% --- red vertical lines at V2 spike times in every panel ---
for k = 1:numel(ax)
    yl = ylim(ax(k));
    hold(ax(k), 'on')
    for s = 1:numel(v2tms)
        plot(ax(k), [v2tms(s) v2tms(s)], yl, ':', 'Color', [1 0 0 0.75], 'LineWidth', 1);
    end
    hold(ax(k), 'off')
end
