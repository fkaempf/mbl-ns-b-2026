% ifr_5min.m
% Firing rate (IFR) in 5-minute bins across all four leech files, both nerves
% (C3 = left, C4 = right). Files are concatenated on one experiment timeline.
% Dopamine was bath-applied 20 min into file 1.
%
% Run from this folder in MATLAB:  ifr_5min
% Needs abfload.m (in the sibling mbl-ns-b-2026/leech folder).

clear; close all;
% Resolve paths relative to THIS script, not MATLAB's current folder.
thisDir = fileparts(mfilename('fullpath'));
if isempty(thisDir), thisDir = pwd; end
addpath(fullfile(thisDir, '..', 'mbl-ns-b-2026', 'leech'));   % abfload

fnames  = {'1.abf','2.abf','3.abf','4.abf'};   % already in recording-time order
files   = fullfile(thisDir, fnames);
binSec  = 300;          % 5-minute bins
kThresh = 4.5;          % spike threshold = k * MAD noise estimate
refrSec = 0.002;        % spike refractory period (s)
DAinFile1 = 20*60;      % dopamine added 20 min (1200 s) into file 1
nCh     = 2;
chNames = {'C3 (left)','C4 (right)'};

allCenters = [];        % bin centers on global timeline (s)
allRate    = [];        % rows = bins, cols = channels  (Hz)
fileEdges  = [];        % file boundaries on global timeline (s)
offset     = 0;         % cumulative experiment time (s)
DAglobal   = NaN;

for f = 1:numel(files)
    [d, si] = abfload(files{f});      % d = samples x channels, si in microseconds
    fs   = 1e6 / si;
    dur  = size(d,1) / fs;

    % 5-min bin edges (clamp the last partial bin to the true file end)
    edges = 0:binSec:dur;
    if edges(end) < dur, edges = [edges dur]; end
    widths  = diff(edges);
    centers = edges(1:end-1) + widths/2;
    rb = nan(numel(centers), nCh);

    for c = 1:nCh
        x = double(d(:,c));
        % --- threshold-crossing spike detection on |x| ---
        noise = median(abs(x)) / 0.6745;
        thr   = kThresh * noise;
        over  = abs(x) > thr;
        cross = find(over(2:end) & ~over(1:end-1)) + 1;
        % --- enforce refractory period ---
        spk = [];
        if ~isempty(cross)
            refr = round(refrSec * fs);
            keep = cross(1);
            for i = 2:numel(cross)
                if cross(i) - keep(end) >= refr
                    keep(end+1) = cross(i); %#ok<AGROW>
                end
            end
            spk = keep / fs;            % spike times within file (s)
        end
        cnt = histcounts(spk, edges);   % spikes per bin
        rb(:,c) = cnt(:) ./ widths(:);  % -> firing rate (Hz)
    end

    allCenters = [allCenters; offset + centers(:)]; %#ok<AGROW>
    allRate    = [allRate; rb];                     %#ok<AGROW>
    if f == 1, DAglobal = offset + DAinFile1; end
    offset = offset + dur;
    fileEdges(end+1) = offset;                       %#ok<AGROW>
    fprintf('%s: %.1f min, fs=%g Hz\n', fnames{f}, dur/60, fs);
end

% ----------------------------- plot -----------------------------
tmin = allCenters / 60;                 % experiment time in minutes
figure('Color','w','Position',[100 100 1000 450]);
plot(tmin, allRate(:,1), '-o', 'LineWidth', 1.5, 'MarkerSize', 4); hold on;
plot(tmin, allRate(:,2), '-o', 'LineWidth', 1.5, 'MarkerSize', 4);
yl = ylim;
xline(DAglobal/60, 'r--', 'DA added', 'LineWidth', 1.5, ...
      'LabelVerticalAlignment','top');
for e = fileEdges(1:end-1)
    xline(e/60, 'k:');                  % file boundaries
end
xlabel('experiment time (min)');
ylabel('firing rate (Hz, 5-min bins)');
legend(chNames, 'Location', 'northeast');
title('IFR in 5-min bins across all files (DA bath-applied at 20 min)');
grid on; box off;
set(gca, 'Color', 'w', ...               % white axes background
         'XColor', 'k', 'YColor', 'k', ...   % black axis lines/ticks/labels
         'GridColor', 'k');
set(gcf, 'Color', 'w', 'InvertHardcopy', 'off');   % keep white when saving

figDir = fullfile(thisDir, 'figures');
if ~exist(figDir,'dir'), mkdir(figDir); end
exportgraphics(gcf, fullfile(figDir, 'ifr_5min.png'), 'BackgroundColor', 'white');
fprintf('Saved %s\n', fullfile(figDir, 'ifr_5min.png'));
