function setupGurobi()
    % SETUPGUROBI - Automatically find and add Gurobi to MATLAB path
    
    % Check if Gurobi is already in path
    if ~isempty(which('gurobi'))
        fprintf('Gurobi is already in MATLAB path.\n');
        return;
    end
    
    % Common Gurobi installation locations
    possible_paths = {
        '/opt/gurobi1201/linux64/matlab',
        '/opt/gurobi/linux64/matlab',
        'C:\gurobi1201\win64\matlab',
        'C:\gurobi\win64\matlab',
        '/usr/local/gurobi/linux64/matlab',
        '/Applications/gurobi/linux64/matlab'
    };
    
    % Check for environment variable
    gurobi_home = getenv('GUROBI_HOME');
    if ~isempty(gurobi_home)
        gurobi_matlab_path = fullfile(gurobi_home, 'matlab');
        possible_paths = [gurobi_matlab_path; possible_paths];
    end
    
    % Try to find Gurobi
    found = false;
    for i = 1:length(possible_paths)
        if exist(possible_paths{i}, 'dir')
            addpath(possible_paths{i});
            fprintf('Added Gurobi path: %s\n', possible_paths{i});
            found = true;
            break;
        end
    end
    
    if ~found
        error(['Gurobi not found. Please install Gurobi and set up the path.\n' ...
               'You can manually add it using: addpath(''/path/to/gurobi/matlab'')']);
    end
    
    % Save path for future sessions
    savepath;
    fprintf('Gurobi setup complete. Path saved.\n');
end
