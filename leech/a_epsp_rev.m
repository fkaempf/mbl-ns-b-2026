% ----- Area Calculator -----
clear; clc; close all;
addpath(genpath(pwd))  % ----- Add current directory and subfolders to the path -----

% ----- Prompt user to select an ABF data file -----
datafile = uigetfile({'*.abf'}, 'File Selector');

% ----- Load voltage and time vectors from the second channel of the ABF file -----
[vlt, tms] = loadephys(datafile, 2);

% ----- Determine the number of sweeps (i.e., repeated trials) -----
a = size(vlt);
sweeps = a(3);  % ----- Third dimension is number of sweeps -----

% ----- Generate a color map for plotting (more color gradation for more sweeps) -----
CM = othercolor('Blues9', sweeps * 4);

% ----- Define the time window for EPSP analysis (based on sample indices) -----
[posids, ~] = peakfinder2(diff(vlt(:,4,5)), 3, 2, 1);
idxon=posids(1);
[negids, ~] = peakfinder2(diff(vlt(:,1,1)), 3, 2, 1);
idxoff=negids(end);
tmsepsp = tms(idxon:idxoff);

% ----- Extract and baseline-subtract EPSPs from each sweep -----
for ii = 1:sweeps
    epsp(:,ii) = vlt(idxon:idxoff,1,ii);                % ----- Extract voltage trace -----
    epspadj(:,ii) = epsp(:,ii) - epsp(1,ii);           % ----- Subtract baseline (first point) -----
end

% ----- Compute area under the curve (AUC) for each sweep, then sort -----
auc = sort(trapz(epspadj))'        % ----- Integration across time (columns) -----
auctxt = string(auc);               % ----- Convert numeric AUCs to strings for legend -----

% ----- Flip the EPSP matrix left-to-right for plotting purposes (descending order) -----
epspadj_ = fliplr(epspadj);

%% ----- Plot all adjusted EPSPs (flipped) with color mapping and overlay AUC -----
figs(1) = figure(); hold on;
for ii = 1:sweeps
    plot(tmsepsp, epspadj_(:,ii), 'Color', CM(ii * (sweeps - 1), :))
end
box off;
axis([tmsepsp(1) tmsepsp(end) -2 10])
yline(0, 'k--')
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel 'Adjusted EPSP Amplitude (mV)';
xlabel 'Time (s)';
legend(auctxt)  % ----- Show sorted AUCs -----

% ----- Flip back to original orientation for ordered EPSP plotting -----
epspf = fliplr(epspadj);

%% ----- Plot selected EPSPs individually with shaded area (AUC) highlighting -----
figs(2) = figure();

% ----- Subplot 1: EPSP #5 -----
subplot(5,1,1)
plot(tmsepsp, epspadj(:,5), 'Color', CM(4,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp, epspadj(:,5), 'FaceColor', CM(4,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,5)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 2: EPSP #4 -----
subplot(5,1,2)
plot(tmsepsp, epspadj(:,4), 'Color', CM(8,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp, epspadj(:,4), 'FaceColor', CM(8,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,4)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 3: EPSP #3 -----
subplot(5,1,3)
plot(tmsepsp, epspadj(:,3), 'Color', CM(12,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp, epspadj(:,3), 'FaceColor', CM(12,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,3)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 4: EPSP #2 -----
subplot(5,1,4)
plot(tmsepsp, epspadj(:,2), 'Color', CM(16,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp, epspadj(:,2), 'FaceColor', CM(16,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,2)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 5: EPSP #1 -----
subplot(5,1,5)
plot(tmsepsp, epspadj(:,1), 'Color', CM(20,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp, epspadj(:,1), 'FaceColor', CM(20,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,1)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';
