%% MAIN PROTOCOL FOR ECOMODEL CONSTRUCTION AND ANALYSIS
% This function performs a complete pipeline for constructing and analyzing 
% an enzyme-constrained metabolic model (ecModel) using GECKO.
%
% Parameters (all optional via name-value pairs):
%   project_name: Name of the project directory (string)
%   input_folder: Path to input data folder (string)
%   ec_params: CSV file with model parameters (string)
%   c_source_val: Carbon source uptake rate (numeric, default=0)
%
% Outputs:
%   - Creates project structure with organized files and folders
%   - Generates and exports the final ecModel
%   - Produces kcat and flux variability analysis outputs

%% MAIN PROTOCOL FOR ECOMODEL CONSTRUCTION AND ANALYSIS
function gecko_protocol(varargin)
    % Force output to stay live and redirect to a log file
    diary('matlab_progress.log'); 
    diary on;

    % --- INPUT PARSING AND INITIALIZATION ---
    p = inputParser;
    addParameter(p, 'project_name', '', @ischar);
    addParameter(p, 'input_folder', '', @ischar);
    addParameter(p, 'ec_params', '', @ischar);
    addParameter(p, 'c_source_val', 0, @isnumeric);

    parse(p, varargin{:});
    args = p.Results;

    projectPath = pwd;
    projectName = args.project_name;

    % --- PROJECT SETUP ---
    startGECKOproject(projectName, projectPath);
    
    % Use fprintf with \n for immediate log updates
    fprintf('Project Name: %s\n', projectName);
    fprintf('Project Path: %s\n', projectPath);
    
    projectInputPath = args.input_folder;
    copyfile(fullfile(projectPath, projectInputPath), ...
             fullfile(projectPath, projectName, 'data'), 'f');

    ecParams = args.ec_params;
    adapterLocation = fullfile(projectPath, projectName, ...
                               [projectName 'Adapter.m']);
    modifyAdapterFromCSV(adapterLocation, ...
                         fullfile(projectPath, projectName, 'data', ecParams));
    setupGurobi();
    setRavenSolver('gurobi');

    % --- STAGE 1: MODEL PREPARATION ---
    fprintf('Stage 1 - Start: %s\n', char(datetime('now')));
    
    tic;
    ModelAdapter = ModelAdapterManager.setDefault(adapterLocation);
    params = ModelAdapter.getParameters();
    
    model = loadConventionalGEM();
    
    DB = loadDatabases('kegg');
    fID = fopen(fullfile(params.path, 'data', 'uniprotConversion.tsv'), 'w');
    output = transpose([DB.kegg.keggGene, DB.kegg.uniprot]);
    fprintf(fID, '%s\t%s\n', 'genes', 'uniprot');
    fprintf(fID, '%s\t%s\n', output{:});
    fclose(fID);
    
    [ecModel, ~] = makeEcModel(model, false);
    [ecModel, ~, ~] = applyComplexData(ecModel);
    
    % --- STAGE 1 VALIDATION ---
    fprintf('==== Stage 1: Glucose ====\n');
    sol = solveLP(ecModel, 1);
    bioRxnIdx = getIndexes(ecModel, params.bioRxn, 'rxns');
    fprintf('ecModel Growth rate: %f /hour.\n', sol.x(bioRxnIdx));
    
    elapsedTime = toc;
    fprintf('Stage 1 - Elapsed time: %.2f seconds\n', elapsedTime);

    % --- STAGE 2: KCAT CONSTRAINTS ---
    fprintf('Stage 2 - Start: %s\n', char(datetime('now')));
    
    tic;
    ecModel = getECfromGEM(ecModel);
    noEC = cellfun(@isempty, ecModel.ec.eccodes);
    ecModel = getECfromDatabase(ecModel, noEC);
    ecModel = getECfromDatabase(ecModel);
    
    [ecModel, ~] = findMetSmiles(ecModel);
    writeDLKcatInput(ecModel, [], [], [], [], true);
    runDLKcat(); 
    
    kcatList_DLKcat = readDLKcatOutput(ecModel);
    kcatList_merged = kcatList_DLKcat; 
    
    ecModel = selectKcatValue(ecModel, kcatList_merged);
    [ecModel, ~, ~] = applyCustomKcats(ecModel);
    ecModel = getKcatAcrossIsozymes(ecModel);
    
    [ecModel, ~, ~, ~] = getStandardKcat(ecModel);
    ecModel = applyKcatConstraints(ecModel);
    
    Ptot = params.Ptot;
    f = params.f;
    sigma = params.sigma;
    f = calculateFfactor(ecModel);
    ecModel = setProtPoolSize(ecModel, Ptot, f, sigma);
    
    % --- STAGE 2 VALIDATION ---
    fprintf('==== Stage 2: Glucose ====\n');
    sol = solveLP(ecModel, 1);
    bioRxnIdx = getIndexes(ecModel, params.bioRxn, 'rxns');
    fprintf('ecModel Growth rate: %f /hour.\n', sol.x(bioRxnIdx));
    
    elapsedTime = toc;
    fprintf('Stage 2 - Elapsed time: %.2f seconds\n', elapsedTime);

    % --- STAGE 3: MODEL TUNING ---
    fprintf('Stage 3 - Start: %s\n', char(datetime('now')));
    
    tic;
    c_source_val = args.c_source_val;
    ecModel = setParam(ecModel, 'lb', params.c_source, c_source_val);
    ecModel = setParam(ecModel, 'obj', params.bioRxn, 1);
    
    fprintf('==== Stage 3 (before tuning): Glucose High Flux ====\n');
    sol = solveLP(ecModel, 1);
    bioRxnIdx = getIndexes(ecModel, params.bioRxn, 'rxns');
    fprintf('ecModel Growth rate: %f /hour.\n', sol.x(bioRxnIdx));
    
    ecModel = setProtPoolSize(ecModel);
    [ecModel, ~] = sensitivityTuning(ecModel);
    
    fprintf('==== Stage 3 (after tuning): Glucose High Flux ====\n');
    sol = solveLP(ecModel, 1);
    bioRxnIdx = getIndexes(ecModel, params.bioRxn, 'rxns');
    fprintf('ecModel Growth rate: %f /hour.\n', sol.x(bioRxnIdx));
    
    elapsedTime = toc;
    fprintf('Total Tuning - Elapsed time: %.2f seconds\n', elapsedTime);

    % --- OUTPUT GENERATION ---
    exportModel(ecModel, fullfile(projectPath, projectName, 'models', ...
                                  [projectName '.xml']));

    dataTable = table(...
        kcatList_merged.rxns, ...
        kcatList_merged.genes, ...
        kcatList_merged.kcats, ...
        kcatList_merged.substrates, ...
        'VariableNames', {'Reaction', 'Gene', 'kcat', 'Substrates'});
    writetable(dataTable, ...
               fullfile(projectPath, projectName, 'output', 'kcats.csv'), ...
               'Delimiter', ';');

    minFlux = zeros(numel(model.rxns), 2);
    maxFlux = minFlux;
    
    [minFlux(:,1), maxFlux(:,1)] = ecFVA(model, model);
    [minFlux(:,2), maxFlux(:,2)] = ecFVA(ecModel, model);
    
    output = [model.rxns, model.rxnNames, ...
              num2cell([minFlux(:,1), maxFlux(:,1), ...
                        minFlux(:,2), maxFlux(:,2)])]';
    
    fID = fopen(fullfile(params.path, 'output', 'ecFVA.csv'), 'w');
    fprintf(fID, 'rxnIDs;rxnNames;minFlux;maxFlux;ec-minFlux;ec-maxFlux\n');
    fprintf(fID, '%s;%s;%g;%g;%g;%g\n', output{:});
    fclose(fID);

    diary off;
end