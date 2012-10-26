local eextest = {}
local precision = 1e-5
local mytester

local function template_SpatialSAD(type,dw,dh,runs,perf_factor)
   local default_type = torch.getdefaulttensortype()
   torch.setdefaulttensortype(type)
   local input = image.lena()
   local runs = runs or 5
   local perf_factor = perf_factor or 1.5
   local kn,kp,kh,kw
   kn=100;kp=3;kh=9;kw=9
   local kernels = torch.Tensor(kn,kp,kh,kw) 
   for i = 1,kn do
      kernels[i] = input[{{},{i,i+kh-1},{i,i+kw-1}}]
   end
   local s_module = nn.SpatialSAD(input:size(1),kn,kw,kh,dw,dh)
   s_module:templates(kernels)
   local c_module = nn.SpatialConvolution(input:size(1),kn,kw,kh,dw,dh)
   c_module.weight:copy(kernels)
   local s_time = 0
   local c_time = 0
   for i = 1,runs do
      time = sys.clock()
      s_module:forward(input)
      s_time = s_time + sys.clock() - time
      time = sys.clock()
      c_module:forward(input)
      c_time = c_time + sys.clock() - time
   end
   ratio = s_time / c_time
   print()
   print('SpatialSAD takes ' .. ratio .. ' times as long as SpatialConv on ' .. type)
   mytester:assertlt(ratio,perf_factor,' - performance err (type: ' .. type .. ')')
   torch.setdefaulttensortype(default_type)
end

function eextest.SpatialSAD_1() template_SpatialSAD('torch.FloatTensor',1,1) end
function eextest.SpatialSAD_2() template_SpatialSAD('torch.DoubleTensor',1,1) end
function eextest.SpatialSAD_3() template_SpatialSAD('torch.FloatTensor',4,1,5,2) end
function eextest.SpatialSAD_4() template_SpatialSAD('torch.DoubleTensor',4,1,5,2) end

-- SpatialSAD testing:
-- TODO: add Jacobian test once derivatives are implemented
-- TODO: add tests for batch version once implemented

function eex.test_performance(tests)
   xlua.require('image',true)
   xlua.require('sys',true)
   mytester = torch.Tester()
   mytester:add(eextest)
   torch.manualSeed(os.time())
   mytester:run(tests)
end
