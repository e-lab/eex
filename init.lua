require 'torch'
require 'xlua'
require 'nn'

-- create global eex table:
eex = {}

-- c lib:
require 'libeex'

-- nn modules:
torch.include('eex', 'SpatialSAD.lua')

-- testing:
torch.include('eex', 'test-modules.lua')
torch.include('eex', 'test-performance.lua')
