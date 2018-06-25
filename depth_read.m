function depth = depth_read(filename)
% Reads a depth file FILENAME into depth image DEPTH. 

% Adapted from Deqing Sun and Daniel Scharstein's optical
% flow code.

% Copyright (c) 2015 Jonas Wulff
% Max Planck Institute for Intelligent Systems, Tuebingen, Germany.


TAG_FLOAT = 202021.25;  % check for this when READING the file

% sanity check
if isempty(filename) == 1
    error('depth_read: empty filename');
end;

idx = findstr(filename, '.');
idx = idx(end);

if length(filename(idx:end)) == 1
    error('depth_read: extension required in filename %s', filename);
end;

if strcmp(filename(idx:end), '.dpt') ~= 1    
    error('depth_read: filename %s should have extension ''.dpt''', filename);
end;

fid = fopen(filename, 'r');
if (fid < 0)
    error('depth_read: could not open %s', filename);
end;

tag     = fread(fid, 1, 'float32');
width   = fread(fid, 1, 'int32');
height  = fread(fid, 1, 'int32');

% sanity check

if (tag ~= TAG_FLOAT)
    error('depth_read(%s): wrong tag (possibly due to big-endian machine?)', filename);
end;

if (width < 1 || width > 99999)
    error('depth_read(%s): illegal width %d', filename, width);
end;

if (height < 1 || height > 99999)
    error('depth_read(%s): illegal height %d', filename, height);
end;

% arrange into matrix form
tmp = fread(fid, inf, 'float32');
depth = reshape(tmp, [width, height])';

fclose(fid);

