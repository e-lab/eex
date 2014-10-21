require 'torch'
require 'xlua'
require 'nn'
require 'sys'

-- create global eex table:
eex = {}

-- c lib:
require 'libeex'

-- nn modules:
torch.include('eex', 'SpatialSAD.lua')
torch.include('eex', 'SpatialSADMap.lua')
torch.include('eex', 'SpatialMaxMap.lua')
torch.include('eex', 'MulAnySize.lua')
torch.include('eex', 'GaborLayer.lua')

-- testing:
torch.include('eex', 'test-modules.lua')
torch.include('eex', 'test-performance.lua')

-- useful functions:
torch.include('eex', 'usefulFunctions.lua')
