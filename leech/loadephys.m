function [dat,tms,h] = loadephys(fn,type)
% LOADEPHYS - Load electrophysiology traces
%    dat = LOADEPHYS(fn) loads the electrophysiology file FN and returns
%    the data as DAT as an LxN array, where L is the length of the recording
%    and N is the number of channels recorded.
%    [dat,tms] = LOADEPHYS(fn,type) returns a vector of time stamps as well.
% TYPE - enter a number which signifies the type of file you want to
% important:
%       1: pClamp files older than 10.4
%       2: pClamp files in 10.4
%       3: pClamp files in version 10.5 or older

if ~exist(fn)
    error(sprintf('LOADEPHYS: File "%s" not found\n',fn));
end

idx = find(fn == '.');
if isempty(idx)
    ext='';
    fnbase=fn;
else
    idx=idx(end);
    ext=fn(idx+1:end);
    fnbase=fn(1:idx-1);
end

switch ext
    case 'abf'
        % type=1; < 10.4 pClamp10 file
        if type==1
            [dat, aux, h] = readabf(fn);
            len=size(dat,1);
            if nargout>=2
                tms=(1:len)'*aux.dt_s;
            end
        elseif type==2; % type=1; >10.4 pClamp10 file
            [dat, si, h] = abfload(fn);
            len=size(dat,1);
            if nargout>=2
                tms=(1:len)'*(si*1e-6);
            end
        elseif type==3; % type=1; >10.5 pClamp10 file
            [dat, si, h] = abfload(fn);
            len=size(dat,1);
            if nargout>=2
                tms=(1:len)'*(si*1e-6);
            end
        end
        
        skp1=std(diff(dat(1:2:500,1)));
        skp2=std(diff(dat(2:2:500,1)));
        skp=std(diff(dat(1:500,1)));
        if type==2
            %             if skp1<skp && skp2<skp
            %                 % This must be interleaved min/max data
            %                 fprintf(1,'Assuming min/max data; returning only greatest absolute values\n');
            %                 C=size(dat,2);
            %                 for c=1:C
            %                     mn=dat(1:2:end,c);
            %                     mx=dat(2:2:end,c);
            %                     usemx=abs(mx)>abs(mn);
            %                     mn(usemx) = mx(usemx);
            %                     dat(1:2:end,c) = mn;
            %                 end
            %                 dat=dat(1:2:end,:);
            %                 if nargout>=2
            %                     tms=tms(1:2:end);
            %                 end
            %             end
        end
    case 'daq'
        % Matlab DAQ file
        if type==4
            if nargout>=2
                if exist('daqread')
                    [dat,tms]=daqread(fn);
                else
                    error('This version of matlab/octave does not support DAQREAD');
                end
            else
                dat = daqread(fn);
            end
        end
    case 'xml'
        % vsdscope / vscope file
        dat = vsdload(fn);
        if nargout>=2
            tms=[1:size(dat.analog.dat,1)]' / dat.analog.info.rate_hz;
        end
        dat = dat.analog.dat;
    case 'escope'
        % Python EScope file
        fd = fopen([fnbase '.txt']);
        clear str
        while 1
            txt = fgets(fd);
            if ischar(txt)
                while txt(end)<' '
                    txt=txt(1:end-1);
                end
                eval(['str.' txt ';']);
            else
                break
            end
        end
        fclose(fd);
        fd = fopen([fnbase '.dat']);
        dat = fread(fd,[str.nchannels,inf],'double')';
        fclose(fd);
        if nargout>=2
            tms=[1:size(dat,1)]'/str.rate_hz;
        end
    otherwise
        error('loadephys: Unknown file format');
end