require 'torch'
require 'xlua'
require 'nn'

-- create global eex table:
eex = {}

-- c lib:
require 'libeex'

-- nn modules:
torch.include('eex', 'SpatialSAD.lua')
torch.include('eex', 'SpatialSADMap.lua')
torch.include('eex', 'SpatialMaxMap.lua')
torch.include('eex', 'MulAnySize.lua')

-- testing:
torch.include('eex', 'test-modules.lua')
torch.include('eex', 'test-performance.lua')
