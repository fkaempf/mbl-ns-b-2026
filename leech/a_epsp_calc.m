% ----- Area Calculator -----
clear; clc; close all;
addpath(genpath(pwd))  % ----- Add current directory and subfolders to the path -----

% ----- Prompt user to select an ABF data file -----
% datafile = uigetfile({'*.abf'}, 'File Selector');
freqs=[5, 10, 20, 40];

% ----- Load voltage and time vectors from the second channel of the ABF file -----
[vlt, tms] = loadephys('P_AP_2.abf', 2);

% ----- Determine the number of sweeps (i.e., repeated trials) -----
a = size(vlt);
channels = 1:a(2);  % ----- Third dimension is number of sweeps -----
sweeps = a(3);

% ----- Generate a color map for plotting (more color gradation for more sweeps) -----
CM = othercolor('Blues9', length(freqs) * 4);

%% ----- Define the time window for EPSP analysis (based on sample indices) -----
[posids, ~] = peakfinder2(diff(vlt(:,3)), 3, 1, 1);
[negids, ~] = peakfinder2(diff(vlt(:,3)), 3, -1, 1);
spktimes=tms(posids);
timeGaps=find(diff(spktimes)>1);
startids=[posids(1);posids(timeGaps+1)];
endids=[negids(timeGaps);negids(end)];
startTimes=tms(startids);

%% ----- Extract and baseline-subtract EPSPs from each sweep -----
epsp_end=vlt(startids(end):end,1);
tms_end=tms(startids(end):end);
adj_end=epsp_end-epsp_end(1);
epspadj(1:length(adj_end),length(freqs))=adj_end;
tmsepsp(1:length(tms_end),length(freqs))=tms_end;
auc(length(freqs),1)=trapz(adj_end)
epsplen=length(epspadj);
for ii = 1:length(freqs)-1
    epsp_ = vlt(startids(ii):startids(ii+1), 1);  % Extract voltage trace
    tmsepsp_=tms(startids(ii):startids(ii+1));
    adj = epsp_ - epsp_(1);
    auc(ii)=trapz(adj)% Adjusted trace
    adjlen=length(adj);
    padlen=epsplen-adjlen;
    epspadj(:,ii) = [adj;NaN(padlen,1)];
    tmsepsp(:,ii)=[tmsepsp_;NaN(padlen,1)];% Store adjusted
end
%% ----- Compute area under the curve (AUC) for each sweep, then sort -----
% auc = sort(trapz(epspadj))'        % ----- Integration across time (columns) -----
auctxt = string(auc);               % ----- Convert numeric AUCs to strings for legend -----

% ----- Flip the EPSP matrix left-to-right for plotting purposes (descending order) -----
epspadj_ = fliplr(epspadj);

%% ----- Plot all adjusted EPSPs (flipped) with color mapping and overlay AUC -----

scrn = get(0, 'ScreenSize');
figWidth = scrn(3) * 0.6;
figHeight = scrn(4) * 0.6;
figLeft = (scrn(3) - figWidth) / 2;
figBottom = (scrn(4) - figHeight) / 2;


figs(1) = figure('Position', [figLeft figBottom figWidth figHeight]); hold on;
for ii = 1:sweeps
    plot(tmsepsp(:,ii)-tmsepsp(1,ii),epspadj_(:,ii))
end
box off;
axis([0 tms(length(tmsepsp)) -2 10])
yline(0, 'k--')
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel 'Adjusted EPSP Amplitude (mV)';
xlabel 'Time (s)';
legend(auctxt)  % ----- Show sorted AUCs -----

% ----- Flip back to original orientation for ordered EPSP plotting -----
epspf = fliplr(epspadj);

% ----- Plot selected EPSPs individually with shaded area (AUC) highlighting -----
figs(2) = figure('Position', [figLeft figBottom figWidth figHeight]); hold on;

% ----- Subplot 1: EPSP #5 -----
% subplot(5,1,1)
% plot(tmsepsp, epspadj(:,1), 'Color', CM(4,:), 'Linewidth', 2); hold on;
% yline(0, 'k--');
% area(tmsepsp, epspadj(:,1), 'FaceColor', CM(4,:), 'Linewidth', 2)
% box off; axis([-inf inf min(epspadj(:,5)) - 3, 10])
% set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
% ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 2: EPSP #4 -----
subplot(4,1,1)
plot(tmsepsp(:,4)-tmsepsp(1,4), epspadj(:,4), 'Color', CM(8,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp(:,4)-tmsepsp(1,4), epspadj(:,4), 'FaceColor', CM(8,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,4)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 3: EPSP #3 -----
subplot(4,1,2)
plot(tmsepsp(:,3)-tmsepsp(1,3), epspadj(:,3), 'Color', CM(8,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp(:,3)-tmsepsp(1,3), epspadj(:,3), 'FaceColor', CM(8,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,4)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 4: EPSP #2 -----
subplot(4,1,3)
plot(tmsepsp(:,2)-tmsepsp(1,2), epspadj(:,2), 'Color', CM(8,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp(:,2)-tmsepsp(1,2), epspadj(:,2), 'FaceColor', CM(8,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,4)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

% ----- Subplot 5: EPSP #1 -----
subplot(4,1,4)
plot(tmsepsp(:,1)-tmsepsp(1,1), epspadj(:,1), 'Color', CM(8,:), 'Linewidth', 2); hold on;
yline(0, 'k--');
area(tmsepsp(:,1)-tmsepsp(1,1), epspadj(:,1), 'FaceColor', CM(8,:), 'Linewidth', 2)
box off; axis([-inf inf min(epspadj(:,4)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';

figs(3)=figure('Position', [figLeft figBottom figWidth figHeight]); hold on;
plot(tms,vlt(:,1,1))
box off; 
% axis([-inf inf min(epspadj(:,4)) - 3, 10])
set(gca, 'FontName', 'Arial', 'FontSize', 16, 'linewidth', 1, 'TickDir', 'out')
ylabel({'EPSP';'Ampl (mV)'}); xlabel 'Time (s)';