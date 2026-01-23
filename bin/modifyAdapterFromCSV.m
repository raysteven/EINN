function modifyAdapterFromCSV(adapterFile, csvFile)
    % MODIFYADAPTERFROMCSV - Modify adapter parameters using a CSV file
    %
    % Usage:
    %   modifyAdapterFromCSV('myProjectNameAdapter.m', 'adapter_params.csv')
    %
    % CSV Format:
    %   Parameter,Value,Type,Description
    %   sigma,0.7,numeric,Average enzyme saturation factor
    %   org_name,Saccharomyces cerevisiae,string,Organism scientific name
    %   ...
    
    % Read the CSV file
    try
        paramTable = readtable(csvFile, 'TextType', 'string');
    catch ME
        error('Could not read CSV file: %s\nError: %s', csvFile, ME.message);
    end
    
    % Validate CSV format
    requiredCols = {'Parameter', 'Value', 'Type'};
    if ~all(ismember(requiredCols, paramTable.Properties.VariableNames))
        error('CSV must contain columns: %s', strjoin(requiredCols, ', '));
    end
    
    % Read the adapter file
    fileID = fopen(adapterFile, 'r');
    if fileID == -1
        error('Could not open file: %s', adapterFile);
    end
    
    lines = {};
    lineNum = 1;
    while ~feof(fileID)
        lines{lineNum} = fgetl(fileID);
        lineNum = lineNum + 1;
    end
    fclose(fileID);
    
    % Process each parameter from CSV
    fprintf('Updating parameters from %s:\n', csvFile);
    fprintf('%-20s %-15s %-s\n', 'Parameter', 'New Value', 'Status');
    fprintf('%s\n', repmat('-', 1, 60));
    
    for i = 1:height(paramTable)
        paramName = char(paramTable.Parameter(i));
        paramValue = char(paramTable.Value(i));
        paramType = char(paramTable.Type(i));
        
        % Skip empty rows
        if isempty(paramName) || strcmp(paramName, '')
            continue;
        end
        
        % Process the parameter value based on type
        [formattedValue, success] = formatParameterValue(paramValue, paramType);
        
        if ~success
            fprintf('%-20s %-15s %-s\n', paramName, paramValue, 'SKIPPED (Invalid type)');
            continue;
        end
        
        % Find and update the parameter in the file
        updated = false;
        for j = 1:length(lines)
            line = lines{j};
            
            % Handle nested parameters (like kegg.ID, uniprot.type, etc.)
            if contains(paramName, '.')
                pattern = sprintf('obj\\.params\\.%s\\s*=', regexptranslate('escape', paramName));
            else
                pattern = sprintf('obj\\.params\\.%s\\s*=', paramName);
            end
            
            if ~isempty(regexp(line, pattern, 'once'))
                % Extract indentation
                indent = regexp(line, '^\s*', 'match');
                if isempty(indent)
                    indent = '';
                else
                    indent = indent{1};
                end
                
                % Create the new line
                lines{j} = sprintf('%sobj.params.%s = %s;', indent, paramName, formattedValue);
                updated = true;
                fprintf('%-20s %-15s %-s\n', paramName, paramValue, 'UPDATED');
                break;
            end
        end
        
        if ~updated
            fprintf('%-20s %-15s %-s\n', paramName, paramValue, 'NOT FOUND');
        end
    end
    
    % Write the modified content back to file
    fileID = fopen(adapterFile, 'w');
    if fileID == -1
        error('Could not write to file: %s', adapterFile);
    end
    
    for i = 1:length(lines)
        fprintf(fileID, '%s\n', lines{i});
    end
    fclose(fileID);
    
    fprintf('\nSuccessfully updated %s\n', adapterFile);
end

function [formattedValue, success] = formatParameterValue(value, type)
    % FORMAT parameter value based on its type
    success = true;
    
    switch lower(type)
        case 'numeric'
            numValue = str2double(value);
            if isnan(numValue)
                success = false;
                formattedValue = value;
            else
                formattedValue = num2str(numValue);
            end
            
        case 'string'
            formattedValue = sprintf("'%s'", value);
            
        case 'logical'
            if strcmpi(value, 'true') || strcmp(value, '1')
                formattedValue = 'true';
            elseif strcmpi(value, 'false') || strcmp(value, '0')
                formattedValue = 'false';
            else
                success = false;
                formattedValue = value;
            end
            
        case 'path'
            % Handle paths - assume they need fullfile if they contain a comma
            if contains(value, ',')
                pathParts = strsplit(value, ',');
                pathParts = strtrim(pathParts); % Remove whitespace
                if length(pathParts) == 2
                    formattedValue = sprintf("fullfile('%s', '%s')", pathParts{1}, pathParts{2});
                else
                    formattedValue = sprintf("fullfile(%s)", strjoin(cellfun(@(x) sprintf("'%s'", x), pathParts, 'UniformOutput', false), ', '));
                end
            else
                formattedValue = sprintf("'%s'", value);
            end
            
        case 'array'
            % Handle arrays - expect comma-separated values in brackets or just comma-separated
            value = strtrim(value);
            if startsWith(value, '[') && endsWith(value, ']')
                formattedValue = value; % Already formatted as array
            else
                % Split by comma and create array
                parts = strsplit(value, ',');
                parts = strtrim(parts);
                % Try to determine if numeric or string array
                if all(~isnan(str2double(parts)))
                    formattedValue = sprintf('[%s]', strjoin(parts, ', '));
                else
                    formattedValue = sprintf("{%s}", strjoin(cellfun(@(x) sprintf("'%s'", x), parts, 'UniformOutput', false), ', '));
                end
            end
            
        otherwise
            % Default to string if type is unknown
            formattedValue = sprintf("'%s'", value);
    end
end