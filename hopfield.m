    
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         % %                                             % %
         % %   Computation and Cognition - Exercise #6   % %
         % %    Hopfield Network - Numeric Simulation    % %
         % %                                             % %
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;clc;                                                              
%% Init Hopfield Network
N = 200;
time_steps = 100;
iterations = 50;

% Choose 3 random memories
Xi = sign(randn(N,3));

% Init weight matrix according to hopfield's rule
J = Xi*Xi'/N;
J = J - diag(diag(J)); % set the diagonal elements to 0

% choose memory to corrupt
mem2Corrupt = Xi(:,1);

%% get avg overlaps for q=0.1
q = 0.1;
overlaps = getAverageHopfield(J, 0.1, mem2Corrupt, Xi, N, iterations);
t = table([1:size(Xi,2)]', overlaps, 'VariableNames',{'Memory', sprintf('Average Overlap (q=%.2f, %d iterations)', q, iterations)});
disp(t);

%% get avg overlaps for many q's
q = [0 0.1 0.2 0.3 0.35 0.36 0.37 0.38 0.39 0.4 0.45 0.5];
overlaps = zeros(3, length(q));
for k=1:length(q)
    overlaps(:,k) = getAverageHopfield(J, q(k), mem2Corrupt, Xi, N, iterations);
end

%% get energy over time for 1 trial run
[~, E] = hopfield(corrupt(mem2Corrupt, q(2), N), J);

%% plot results

% overlap as function of q
figure;
plot(q, overlaps);
title("Hopfield final overlap as a function of bits flipped");
subtitle("Each overlap is averaged over " + iterations + " runs");
xlabel("Percentage Bits Flipped");
ylabel("Final Overlap");
legend("Corrupted Memory","Memory 2","Memory 3", "Location", "northwest");

% energy over time
figure;
plot(E);
title("Hopfield network energy over time");
xlabel("Time");
ylabel("Energy");

function [overlaps, E] = getAverageHopfield(J, q, mem2Corrupt, Xi, N, it)
    overlaps = zeros(3,1);
    E = 0;
    for i=1:it
        finalState = hopfield(corrupt(mem2Corrupt, q, N), J);
        overlaps = overlaps + Xi'*finalState/N/it;
    end
end

function corrupted = corrupt(mem, q, N)
    corrupted = sign(unifrnd(-q,1-q, N, 1)).*mem;
end

function [finalState, E] = hopfield(init_state, J)
    finalState = init_state;
    newState = init_state;
    t = 1;
    while true
        for i=1:size(J,1)
            newState(i) = sign(J(i,:)*newState);
            E(t) = -0.5*newState'*J*newState;
            t = t+1;
        end
        % as there is no noise, if the new state is equal to the previous
        % state we can consider the network converged.
        if isequal(newState, finalState)
            break;
        end
        finalState = newState;
    end
end


