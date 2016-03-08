function NBodyLogging(logFile, simType, simSolver, algType, numBody, startTime, ...
    endTime, stepTime, seed, eps, totalTime, avgTime)
    % Date Hour:Min:Sec   -   SIMTYPE  -  SOLVER ALGTYPE
    %       numBody      = 32
    %       interval     = [0, 365.25] [D]
    %       stepsize     = 1 [D]
    %       seed         = 0
    %       eps          = 0.00001 [AU]
    %       ----------------------------
    %       Total time   = 10 [s]
    %       Average time = 10 [s]
    % #########################################################
    fileID = fopen(logFile, 'a');
    
    c = fix(clock);
    
    fprintf(fileID, '%d.%d.%d %d:%d:%d',c);
    fprintf(fileID, '    %s  -  %s %s\n', simType, simSolver, algType);
    fprintf(fileID, '\tnumBody\t\t= %d\n', numBody);
    fprintf(fileID, '\tinterval\t= [%d %d] [D]\n', startTime, endTime);
    fprintf(fileID, '\tstepTime\t= %d [D]\n', stepTime);
    fprintf(fileID, '\tseed\t\t= %d\n', seed);
    fprintf(fileID, '\teps\t\t\t= %d [AU]\n', eps);
    fprintf(fileID, '\t----------------------------\n');
    fprintf(fileID, '\tTotal time\t= %d [s]\n', totalTime);
    fprintf(fileID, '\tAvg time\t= %d [s]\n', avgTime);
    fprintf(fileID, '#####################################################\n');
    
    fclose('all');
end