function run_lifecycle(config_json)
% Run Laibson et al.'s LifecycleSim.m, unmodified, under Octave (RESULTS 54).
%
% The audit compares their code as run, an independent reimplementation
% (src/hh_npe/simulator/reference.py) and our port. This driver sets up
% exactly what their EDFbatch_baseline.m sets up (comphs, benchmark returns,
% naive beta-hat = 1, alpha = 0.5, population 10,000), with these overrides
% read from a JSON file:
%
%   prefs    [beta delta rho]                    (required)
%   out      path of the .mat to write            (required)
%   xjump, xmax, zjump, zmax                      their grid knobs (their
%            50 and 10 cells per step are hard-coded and stay so)
%   R_gamma  illiquid return (A2 estimates it; theirs is 1.05)
%   card     0 = no credit line (A2's no-card type): credit coefficients 0
%   pop      simulated households
%
% Writes their policy arrays, the forward panels, simMoments, the grids, the
% income setup, and the two uniform arrays their forward pass drew (their
% rand('seed', 1) stream, regenerated in the same order), so any solver's
% policies can be run on the same shocks.
%
% Usage (from the repository root):
%   octave --no-gui --quiet --eval "run_lifecycle('cfg.json')"
% with scripts/laibson_octave on the path, or via scripts/audit_simulation.py.

  cfg = jsondecode(fileread(config_json));
  here = fileparts(mfilename('fullpath'));
  root = fileparts(fileparts(here));
  out_path = cfg.out;
  if out_path(1) ~= '/'
    out_path = fullfile(root, out_path);
  end
  pkg load statistics
  warning('off', 'all');
  addpath(fullfile(here, 'shim'));
  code = fullfile(root, 'replication-package-LLMRT', 'LifecycleSimulation', 'code');
  addpath(fullfile(code, 'main'));
  old = cd(code);              % FirstStageParams reads ../input relative to here
  cleanup = onCleanup(@() cd(old));

  % The shim's tie and extrapolation behaviour, checked rather than assumed.
  F = griddedInterpolant([1 2 3], [1 2 3], 'nearest');
  assert(isequal(F([2.5; -7; 9]), [3; 1; 3]));

  % --- EDFbatch_baseline.m's setup, comphs ---------------------------------
  setup.educ = 'comphs';
  setup.pop = 10000;
  setup.xnonlin = 1; setup.xjump = 1000; setup.xmax = 4e5;
  setup.znonlin = 1; setup.zjump = 2000; setup.zmax = 3.5e6;
  setup.naif = 1; setup.betahat = 1;
  setup.alpha = 0.5; setup.zilliq = 0; setup.sqrtscale = 0;
  setup.mortalityn = 0; setup.initwseed = 0;
  setup.split_income_at_retirement = 0;
  setup.income_var_multiplier = [1 1];
  setup.income_auto9 = 0; setup.income_auto9revall = 0;
  setup.intrates = 'benchmark';
  setup.returns = [1.0203 1.0500 1.1059];

  % --- overrides ------------------------------------------------------------
  for f = {'xjump', 'xmax', 'zjump', 'zmax', 'pop'}
    if isfield(cfg, f{1})
      setup.(f{1}) = cfg.(f{1});
    end
  end
  if isfield(cfg, 'R_gamma')
    setup.returns(2) = cfg.R_gamma;
  end
  fs_params = FirstStageParams(setup);
  if isfield(cfg, 'card') && cfg.card == 0
    fs_params.c0credit = 0; fs_params.c1credit = 0; fs_params.c2credit = 0;
  end

  t0 = time();
  out = LifecycleSim(reshape(cfg.prefs, 1, 3), setup, fs_params);
  elapsed = time() - t0;

  % Their forward pass drew rand(pop, T) twice right after rand('seed', 1)
  % (LifecycleSim.m:19, LifecycleSim_ForwardIter.m:74-75); the backward
  % induction draws nothing, so this is the same stream.
  T = out.income_setup.yearsalive;
  rand('seed', 1);
  randU = rand(setup.pop, T);
  randY = rand(setup.pop, T);

  bw = out.bwInductOut; fw = out.fwIterOut; g = out.grid_setup; inc = out.income_setup;
  Ix = bw.Ix____; Iz = bw.Iz____; C = bw.C____; V = bw.V____; EV = bw.EV____; W = bw.W____;
  simY = fw.simY__; simX = fw.simX__; simZ = fw.simZ__; simC = fw.simC__;
  simXind = fw.simXind__; simZind = fw.simZind__; simUind = fw.simUind__;
  simMoments = out.simMoments;
  X = g.X__; Z = g.Z_; nX = g.nX_; xmin = g.xmin_;
  ymean = inc.ymean_; Ymean = inc.Ymean_; effhhN = inc.effhhN_; zliqpen = inc.zliqpen_;
  xiS = inc.xiS__(:, :, 1); xiP = inc.xiP___(:, :, 1); death = inc.death_;
  prefs = reshape(cfg.prefs, 1, 3); returns = setup.returns;
  save('-v7', out_path, 'Ix', 'Iz', 'C', 'V', 'EV', 'W', 'simY', 'simX', 'simZ', ...
       'simC', 'simXind', 'simZind', 'simUind', 'simMoments', 'X', 'Z', 'nX', ...
       'xmin', 'ymean', 'Ymean', 'effhhN', 'zliqpen', 'xiS', 'xiP', 'death', ...
       'randU', 'randY', 'prefs', 'returns', 'elapsed');
  printf('wrote %s in %.0f s\n', out_path, elapsed);
end
