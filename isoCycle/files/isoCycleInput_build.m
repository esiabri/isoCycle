
% this functions converts the output of the kilosort to an npy file that
% can be used as the input for isoCycle

% this function uses npy-matlab module, https://github.com/kwikteam/npy-matlab
% make sure to add this module to your matlab path before running this
% function

% to run this function:
% add this file to your matlab path
% in matlab command line type isoCycleInput_build()

function isoCycleInput_build(allValidClusters)

    % if allValidClusters: True, spikes from both 'good' and 'mua' clusters
    % are included otherwise only the 'good' clusters are included
    
    % Check if allValidClusters is empty
    if nargin < 1
        allValidClusters = true;
    end
    
    % Open file selection window to choose spike_times.npy file
    [fileName, path] = uigetfile('*.npy', 'Select spike_times.npy file');
    
    % Check if the user canceled the file selection
    if isequal(fileName, 0) || isequal(path, 0)
        allValidSpikes = [];
        return;
    end
    
    spikesSampleFileAdd = fullfile(path, 'spike_times.npy');
    spikeClusterFileAdd = fullfile(path, 'spike_clusters.npy');

    paramsFileAdd = fullfile(path, 'params.py');

    % Read the params.py file line by line
    fileID = fopen(paramsFileAdd, 'r');
    lines = textscan(fileID, '%s', 'Delimiter', '\n');
    lines = lines{1};
    fclose(fileID);

    % Initialize sample_rate variable
    sample_rate = [];
    
    % Search for the line containing 'sample_rate' and extract the value
    for i = 1:numel(lines)
        line = lines{i};
        match = regexp(line, 'sample_rate\s*=\s*([\d.]+)', 'tokens');
        if ~isempty(match)
            sample_rate = str2double(match{1}{1});
            break;
        end
    end
    
    % Check if sample_rate was successfully extracted
    if isempty(sample_rate)
        error('Unable to extract sample_rate from params.py');
    end

    spikeClusters = readNPY(spikeClusterFileAdd); % Requires readNPY function (https://github.com/kwikteam/npy-matlab)
    spikesSample = readNPY(spikesSampleFileAdd); % Requires readNPY function

    clusterLabelFileAdd = fullfile(path, 'cluster_info.tsv');

    clusterId = [];
    clusterLabel = [];

    if exist(clusterLabelFileAdd, 'file') == 2
        clusterInfo = tdfread(clusterLabelFileAdd, 'tab');
        clusterId = clusterInfo.id;
        clusterLabel = cellstr(clusterInfo.group);
    else
        clusterLabelFileAdd = fullfile(path, 'cluster_KSLabel.tsv');
        clusterInfo = tdfread(clusterLabelFileAdd, 'tab');
        clusterId = clusterInfo.cluster_id;
        clusterLabel = cellstr(clusterInfo.KSLabel);
    end

    clusterId = clusterId(2:end);
    clusterLabel = clusterLabel(2:end);

    spikesTimes = double(spikesSample) / sample_rate;
    
    % Check if 'mua' clusters exist in clusterLabel array
    muaClustersExist = any(strcmp(clusterLabel, 'mua'));
    if muaClustersExist
        MUA_clusters = clusterId(strcmp(clusterLabel, 'mua'));
    else
        MUA_clusters = [];
    end
    
    % Check if 'good' clusters exist in clusterLabel array
    goodClustersExist = any(strcmp(clusterLabel, 'good'));
    if goodClustersExist
    SUA_clusters = clusterId(strcmp(clusterLabel, 'good'));
    else
        SUA_clusters = [];
    end

    if allValidClusters
        clusterNumbers = [MUA_clusters; SUA_clusters];
    else
        clusterNumbers = SUA_clusters;
    end % Add a closing parenthesis here

    allValidSpikes = [];

    for clusterCounter = 1:numel(clusterNumbers)
        clusterNo = clusterNumbers(clusterCounter);

        clusterSpikeTimes = spikesTimes(find(spikeClusters == clusterNo));

        allValidSpikes = [allValidSpikes; clusterSpikeTimes];
    end

    allValidSpikes = sort(allValidSpikes(:));
    
    allValidSpikes = allValidSpikes(:);

%     save('spikesTimes.mat', 'allValidSpikes');
    writeNPY(allValidSpikes,fullfile(path, 'spikesTimes_isoCycleInput.npy'))
end