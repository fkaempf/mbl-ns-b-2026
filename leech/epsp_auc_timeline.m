xq% epsp_auc_timeline.m
% -------------------------------------------------------------------------
% Open multiple ABF files, order them by their true recording time, and for
% every sweep compute the area under the curve (AUC) of the compound EPSP
% evoked on the postsynaptic cell.  All EPSPs are placed on one continuous
% experiment timeline and plotted as AUC vs time, so synaptic strength can be
% tracked across the whole long-term recording.
%
% Paradigm (verified against the data headers):
%   - Episodic recordings: sweeps of ~5 s, repeating every ~6 s.
%   - Each sweep delivers a short presynaptic spike train on Red_Vm (col 3),
%     which evokes one compound EPSP on the postsynaptic Grn_Vm (col 1).
%   - One EPSP (one AUC) per sweep.
%
% Dependencies (expected on the MATLAB path, as in your other scripts):
%   loadephys.m
% Spike detection and the ABF timestamp reader are self-contained below, so
% peakfinder2 is NOT required.
%
% Run with:  epsp_auc_timeline
% -------------------------------------------------------------------------

clear; clc; close all;
addpath(genpath(pwd))   % current dir + subfolders (for loadephys etc.)

%% ===================== PARAMETERS (edit here) ===========================
epspCh          = 1;        % channel to integrate (Grn_Vm, postsynaptic EPSP)
spikeCh         = 3;        % channel to detect spikes on (Red_Vm, presynaptic)
loadArg         = 2;        % 2nd arg passed to loadephys (matches your scripts)

winDur          = 2.0;      % EPSP integration window after train onset (s)
baselineWin     = 0.10;     % pre-train baseline window (s)
sweepIntervalSec= 6.0;      % sweep start-to-start interval (s); 6 s for this protocol

% --- presynaptic spike detection (on the derivative of spikeCh) ---
spikeDerivThresh= 1.0;      % min sample-to-sample rise (mV) to count as a spike edge
refractorySec   = 0.003;    % min spacing between detected spikes (s)

% --- optional normalization of AUC to the first N sweeps of the experiment ---
baselineSweepsForNorm = 0;  % 0 = off; e.g. 5 = divide all AUCs by mean of first 5

% --- I/O ---
startDir        = '/Users/fkampf/Desktop/2905';   % uigetfile starts here
outDir          = '';       % '' = ask/derive from data folder; else a folder path
% ========================================================================

%% ----- Select ABF files (multi-select) -----
[fnames, fdir] = uigetfile({'*.abf','Axon binary files (*.abf)'}, ...
    'Select ABF files (multi-select allowed)', ...
    fullfile(startDir, filesep), 'MultiSelect', 'on');
if isequal(fnames, 0)
    error('No files selected.');
end
if ischar(fnames), fnames = {fnames}; end          % single selection -> cell
files = fullfile(fdir, fnames);
nFiles = numel(files);

%% ----- Order files by their true recording start time (from ABF header) -----
startDT = NaT(nFiles,1);
for k = 1:nFiles
    startDT(k) = abf2_start_datetime(files{k});
end
[startDT, ord] = sort(startDT);
files  = files(ord);
fnames = fnames(ord);
t0_exp = startDT(1);        % experiment zero = first recording's start

fprintf('Files in recording order:\n');
for k = 1:nFiles
    fprintf('  %2d. %-40s  %s\n', k, fnames{k}, datestr(startDT(k),'HH:MM:SS'));
end

%% ----- Process each file / sweep -----
res = struct('file',{},'fileIdx',{},'sweep',{}, ...
             't_onset_s',{},'t_global_s',{},'auc',{},'nSpikes',{});

for k = 1:nFiles
    fprintf('Loading %s ...\n', fnames{k});
    [vlt, tms] = loadephys(files{k}, loadArg);     % vlt: samples x channels x sweeps
    [~, nCh, nSweep] = size(vlt);
    if nCh < max(epspCh, spikeCh)
        error('File %s has %d channels; need at least %d.', fnames{k}, nCh, max(epspCh,spikeCh));
    end
    dt = mean(diff(tms));                           % sample period (s)
    fs = 1/dt;
    refrac = max(1, round(refractorySec*fs));       % refractory in samples
    nBase  = max(1, round(baselineWin*fs));         % baseline samples
    nWin   = round(winDur*fs);                      % window samples

    % file time offset relative to experiment start
    fileOffset = seconds(startDT(k) - t0_exp);

    onsets = nan(nSweep,1);                         % onset time (s) within each sweep
    nspk   = zeros(nSweep,1);
    for s = 1:nSweep
        pre = vlt(:,spikeCh,s);
        idx = detect_spike_onsets(pre, spikeDerivThresh, refrac);
        nspk(s) = numel(idx);
        if ~isempty(idx)
            onsets(s) = tms(idx(1));
        end
    end

    % sweeps with no detectable presynaptic spike (subthreshold) still have an
    % EPSP -> use the median onset across detected sweeps so they still get an AUC
    medOnset = median(onsets(~isnan(onsets)));
    if isnan(medOnset), medOnset = tms(1); end      % no spikes anywhere -> sweep start
    onsets(isnan(onsets)) = medOnset;

    for s = 1:nSweep
        post = vlt(:,epspCh,s);
        t0   = onsets(s);
        i0   = find(tms >= t0, 1, 'first');
        if isempty(i0), i0 = 1; end

        % baseline = mean postsynaptic voltage just before the train
        b0 = max(1, i0 - nBase);
        baseline = mean(post(b0:max(b0,i0-1)));

        % integration window [t0, t0+winDur], clipped to the sweep
        i1 = min(numel(post), i0 + nWin);
        seg = post(i0:i1) - baseline;
        auc = trapz(tms(i0:i1), seg);               % mV*s

        res(end+1) = struct( ...
            'file', fnames{k}, 'fileIdx', k, 'sweep', s, ...
            't_onset_s', t0, ...
            't_global_s', fileOffset + (s-1)*sweepIntervalSec + t0, ...
            'auc', auc, 'nSpikes', nspk(s)); %#ok<SAGROW>
    end
end

%% ----- Assemble results table -----
T = struct2table(res);
T.t_global_min = T.t_global_s / 60;

if baselineSweepsForNorm > 0 && height(T) >= baselineSweepsForNorm
    base = mean(T.auc(1:baselineSweepsForNorm));
    T.auc_norm = T.auc / base;
    ycol = 'auc_norm';  ylab = 'EPSP AUC (norm. to baseline)';
else
    ycol = 'auc';       ylab = 'EPSP AUC (mV\cdots)';
end
disp(T);

%% ----- Resolve output folder -----
if isempty(outDir)
    cand = fullfile(fdir, 'figures');
    if isfolder(cand), outDir = cand; else, outDir = fdir; end
end
if ~isfolder(outDir), mkdir(outDir); end

%% ----- Main figure: AUC vs experiment time -----
CM = lines(nFiles);
fig1 = figure('Color','w','Name','EPSP AUC over time'); hold on;
for k = 1:nFiles
    m = T.fileIdx == k;
    plot(T.t_global_min(m), T.(ycol)(m), '-o', ...
        'Color', CM(k,:), 'MarkerFaceColor', CM(k,:), 'MarkerSize', 4, ...
        'LineWidth', 1.2, 'DisplayName', fnames{k});
end
% dashed lines at each file boundary
yl = ylim;
for k = 1:nFiles
    xb = seconds(startDT(k) - t0_exp)/60;
    plot([xb xb], yl, 'k:', 'HandleVisibility','off');
end
ylim(yl);
box off; yline(0, 'k--', 'HandleVisibility','off');
set(gca,'FontName','Arial','FontSize',14,'TickDir','out','LineWidth',1);
xlabel('Experiment time (min)'); ylabel(ylab);
legend('Location','best','Interpreter','none');
title('Compound EPSP AUC per sweep');

%% ----- QC figure: one representative sweep per file -----
fig2 = figure('Color','w','Name','QC: detected EPSP window');
nr = ceil(sqrt(nFiles)); nc = ceil(nFiles/nr);
for k = 1:nFiles
    [vlt, tms] = loadephys(files{k}, loadArg);
    nSweep = size(vlt,3);
    % pick first sweep with a detected spike, else sweep 1
    dt = mean(diff(tms)); fs = 1/dt;
    refrac = max(1, round(refractorySec*fs)); nBase = max(1, round(baselineWin*fs));
    pick = 1; t0 = tms(1);
    for s = 1:nSweep
        idx = detect_spike_onsets(vlt(:,spikeCh,s), spikeDerivThresh, refrac);
        if ~isempty(idx), pick = s; t0 = tms(idx(1)); break; end
    end
    post = vlt(:,epspCh,pick);
    i0 = find(tms >= t0,1,'first'); if isempty(i0), i0=1; end
    b0 = max(1,i0-nBase); baseline = mean(post(b0:max(b0,i0-1)));
    i1 = min(numel(post), i0 + round(winDur*fs));

    subplot(nr,nc,k); hold on;
    plot(tms, post, 'Color', CM(k,:));
    yl = ylim;
    patch([tms(i0) tms(i1) tms(i1) tms(i0)], [yl(1) yl(1) yl(2) yl(2)], ...
        CM(k,:), 'FaceAlpha',0.12, 'EdgeColor','none');
    plot([t0 t0], yl, 'r:', 'LineWidth',1);          % detected onset
    yline(baseline, 'k--');                          % baseline
    ylim(yl); box off;
    set(gca,'FontName','Arial','FontSize',11,'TickDir','out','LineWidth',1);
    title(sprintf('%s (sweep %d)', fnames{k}, pick), 'Interpreter','none','FontSize',9);
    xlabel('Time (s)'); ylabel('Grn\_Vm (mV)');
end

%% ----- Save outputs -----
stamp = datestr(t0_exp,'yyyy-mm-dd');
csvPath = fullfile(outDir, sprintf('epsp_auc_timeline_%s.csv', stamp));
writetable(T, csvPath);
savefig(fig1, fullfile(outDir, sprintf('epsp_auc_timeline_%s.fig', stamp)));
exportgraphics(fig1, fullfile(outDir, sprintf('epsp_auc_timeline_%s.png', stamp)), 'Resolution',200);
savefig(fig2, fullfile(outDir, sprintf('epsp_auc_qc_%s.fig', stamp)));
fprintf('\nSaved table  -> %s\n', csvPath);
fprintf('Saved figures-> %s\n', outDir);

%% ======================= local functions ===============================
function idx = detect_spike_onsets(v, derivThresh, refrac)
% Detect fast upward deflections (spike rising edges) in v.
% Returns sample indices of detected onsets, spaced at least `refrac` apart.
    dv = [0; diff(v(:))];
    cand = find(dv > derivThresh);
    idx = [];
    last = -inf;
    for c = cand'
        if c - last >= refrac
            idx(end+1,1) = c; %#ok<AGROW>
            last = c;
        end
    end
end

function dt = abf2_start_datetime(fn)
% Read the true recording start time from an ABF2 header (bytes 16/20).
% Falls back to the file's filesystem date for non-ABF2 files.
    fid = fopen(fn, 'r', 'ieee-le');
    if fid < 0, error('Cannot open %s', fn); end
    sig = fread(fid, 4, '*char')';
    fseek(fid, 16, 'bof');
    d  = fread(fid, 1, 'uint32');     % uFileStartDate  (YYYYMMDD)
    ms = fread(fid, 1, 'uint32');     % uFileStartTimeMS (ms since midnight)
    fclose(fid);
    if strncmp(sig, 'ABF2', 4) && d > 19000000
        y  = floor(d/10000);
        mo = floor(mod(d,10000)/100);
        da = mod(d,100);
        dt = datetime(y, mo, da) + milliseconds(ms);
    else
        info = dir(fn);
        dt = datetime(info.datenum, 'ConvertFrom', 'datenum');
    end
end
