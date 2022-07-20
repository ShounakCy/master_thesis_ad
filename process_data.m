 %% Preprocesss highD dataset%%
    clear;
    clc;

    grid_length=25; grid_width=5; cell_length=8; cell_width=7;

    
    % Other variable dependent on grid.
    grid_cells = grid_length * grid_width;
    grid_cent_location = ceil(grid_length*grid_width*0.5);


    %% 1.Load data 
    dataset_to_use = 10;
    disp('Loading data...')

    for k = 1:dataset_to_use
        if mod(k,2)
            % forward side:
            dataset_name = sprintf('dataset/%02d-fwd.csv', ceil(k/2));
        else
            % backward side:
            dataset_name = sprintf('dataset/%02d-bck.csv', ceil(k/2));
        end
        csv{k}  = readtable(dataset_name);
        %extracting the first 14 columns
        traj{k} = csv{k}{:,1:14};
        %traj dimensions 59597x14
        % Add dataset id at the 1st column
        %size(A) returns a two-element row vector consisting of the number of rows and the number of table variables
        %size(A,dim) returns the length of dimension dim when dim is a positive integer scalar.
        %single(X) converts the values in X to single precision.
        %siz = size(traj{k},1);
        %59597
        %k_one = k*ones(size(traj{k},1),1);
        %all the rows of the 1st col values k times ones
        %singl = [ k*ones(size(traj{k},1),1), traj{k} ];
        traj{k} = single([ k*ones(size(traj{k},1),1), traj{k} ]); 
        % traj dimesnions 59597x15
        % Add dataset id at the 1st column 
        % 59597x15
        % Finally 1:dataset id, 2:Vehicle id, 3:Frame index, 
        %         6:Local X, 7:Local Y, 15:Lane id.
        %         10:v_length, 11:v_Width, 12:v_Class
        %         13:Velocity (feet/s), 14:Acceleration (feet/s2).
        traj{k} = traj{k}(:,[1,2,3,6,7,15,10,11,12,13,14]);
        % Leave space for maneuver labels (2 columns) and grid (grid_cells columns)
        traj{k} = [traj{k}(:,1:6), zeros(size(traj{k},1),2), traj{k}(:,7:11), zeros(size(traj{k},1),grid_cells)];
        
        lane_num = size(unique(traj{k}(:, 6)), 1);
    end

    % Use the vehilce's center as its location
    offset = zeros(1,dataset_to_use);
    for k = 1:dataset_to_use
        a = traj{k}(:,5);
        b = 0.5*traj{k}(:,9);
        traj{k}(:,5) = traj{k}(:,5) - 0.5*traj{k}(:,9);
        offset(k) = min(traj{k}(:,5));
        if offset(k) < 0
            % To make coordinate Y > 0
            traj{k}(:,5) = traj{k}(:,5) - offset(k);
        end
    end


    for ii = 1:dataset_to_use
        tic;
        disp(['Now process dataset ', num2str(ii)])
    
        % Loop on each row.
        t =traj{ii}(:,1);
        len = length(traj{ii}(:,1));
        for k = 1:length(traj{ii}(:,1)) 
            % Refresh the process every 1 mins
            if toc > 60  
                fprintf( 'Dataset-%d: Complete %.3f%% \n', ii, k/length(traj{ii}(:,1))*100 );
                tic;
            end

            dsId = ii;
            vehId = traj{ii}(k,2);
            time = traj{ii}(k,3);
            % Get all rows about this vehId
            vehtraj = traj{ii}(traj{ii}(:,2)==vehId, : );  
        
            % Get the row index of traj at this frame.

            ind = find(vehtraj(:,3)==time);  
            %find(X) returns a vector containing the linear indices of each nonzero element in array X.
            ind = ind(1);
            lane = traj{ii}(k,6);

             % Lateral maneuver in Column 7:
             %So we check for every index the lane number and compare 
             %with the upper boundary and Lower boundary and fill the 
             %lateral maneuver column accordingly

            ub = min(size(vehtraj,1),ind+40);                                %Upper boundary (+40 frame)
            lb = max(1, ind-40);                                             %Lower boundary (-40 frame)
            if vehtraj(ub,6)>vehtraj(ind,6) || vehtraj(ind,6)>vehtraj(lb,6)  %(prepate to turn or stablize after turn)
                traj{ii}(k,7) = 3;   % Turn Right==>3. 
            elseif vehtraj(ub,6)<vehtraj(ind,6) || vehtraj(ind,6)<vehtraj(lb,6)
                traj{ii}(k,7) = 2;   % Turn Left==>2.
            else
                traj{ii}(k,7) = 1;   % Keep lane==>1.
            end

            % Longitudinal maneuver in Column 8:
            ub = min(size(vehtraj,1),ind+50); % Future boundary  (+50 frame)
            lb = max(1, ind-30);              % History boundary (-30 frame)
            if ub==ind || lb ==ind
                traj{ii}(k,8) = 1;   % Normal==>1 
            else 
            %checking the values in the local_y
                vHist = (vehtraj(ind,5)-vehtraj(lb,5))/(ind-lb);
                vFut = (vehtraj(ub,5)-vehtraj(ind,5))/(ub-ind);
                if vFut/vHist <0.85
                    traj{ii}(k,8) = 2; % Brake==> 2
                else
                    traj{ii}(k,8) = 1; % Normal==>1
                end
            end