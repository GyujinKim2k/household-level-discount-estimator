function F = griddedInterpolant(x, v, method)
% Octave shim for MATLAB's griddedInterpolant, only as Laibson et al. call it
% (RESULTS 54): 1-D, 'nearest', with MATLAB's default nearest extrapolation.
% Octave 8 has no griddedInterpolant. interp1(..., 'nearest', 'extrap')
% resolves an exact midpoint to the higher index and clamps outside the grid
% to its end points (checked in scripts/laibson_octave/run_lifecycle.m).
% MATLAB's own tie direction cannot be checked here; it is the one the port
% assumes (twoasset._nearest_index).
  if nargin < 3 || ~strcmp(method, 'nearest')
    error('griddedInterpolant shim: only the ''nearest'' method is provided');
  end
  x = x(:);
  v = v(:);
  F = @(q) reshape(interp1(x, v, q(:), 'nearest', 'extrap'), size(q));
end
