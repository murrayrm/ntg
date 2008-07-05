% vanderpol.m - Matlab file to read NTG output
% RMM, 2 Feb 02 (based on previous version by MBM)
%
% This file reads the output from NTG and uses the spline toolbox
% to compute the trajectories for the status and the inputs

% Define the breakpoints to be used for plotting
More breakpoints for illustration
bps = linspace(0,5,20);
augbps = sort([bps bps bps]);
laugbps = length(augbps);

% Define the spline used for the output
% Note that this much match the setup in vanderpol.c
order1=5;
mult1=3;
knots1=linspace(0,5,3);
augknots1=augknt(knots1,order1,order1-mult1);
colloc1 = spcol(augknots1,order1,augbps);

% Load the output from NTG
load coef1

% Compute the functions defined by the splines
z1=colloc1*coef1';

% Now extract them into more useful variables
% This uses the problem setup described in README

time = bps;

o1 = z1(1:3:laugbps);			% output
o1d = z1(2:3:laugbps);			% first deriv
o1dd = z1(3:3:laugbps);			% second deriv

u1= o1dd + o1 - (1-o1.^2).*o1d;		% input (see README)
