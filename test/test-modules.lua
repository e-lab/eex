local eextest = {}
local precision = 1e-5
local mytester

local function template_SpatialSAD(mtype,dw,dh)
   local default_type = torch.getdefaulttensortype()
   torch.setdefaulttensortype(mtype)
   local input = image.lena()[{{},{200,300},{200,300}}]
   local kn,kp,kh,kw
   kn=2;kp=3;kh=31;kw=31
   local ker1 = input[{{},{50,50+kh},{50,50+kw}}]
   local ker2 = input[{{},{10,10+kh},{10,10+kw}}]
   local kernels = torch.cat(ker1:resize(1,kp,kh,kw),ker2:resize(1,kp,kh,kw),1)
   function luasad(input,kernels,dw,dh)
      local ip, ih, iw; local kn,kh,kw; local op, oh, ow;
      ip = input:size(1); ih = input:size(2); iw = input:size(3)
      kn = kernels:size(1); kh = kernels:size(3); kw = kernels:size(4)
      op = kn; oh = math.floor((ih-kh)/dh+1); ow = math.floor((iw-kw)/dw+1)
      local output = torch.Tensor(op,oh,ow)
      for k = 0,(kn-1) do
         for i = 0,(oh-1) do
            for j = 0,(ow-1) do
               output[{{k+1},{i+1},{j+1}}] = (input[{{},{i*dh+1,i*dh+kh},{j*dw+1,j*dw+kw}}]-kernels[k+1]):abs():sum()
            end
         end
      end
      return output
   end
   local module = nn.SpatialSAD(input:size(1),kn,kw,kh,dw,dh)
   module:templates(kernels)
   local m_output = module:forward(input)
   local l_output = luasad(input,kernels,dw,dh)
   mytester:assertTensorEq(m_output,l_output,1000*precision,' - output err (type: ' .. mtype .. ', dW: ' .. dw .. ', dH: ' .. dh .. ')')
   torch.setdefaulttensortype(default_type)
end
function eextest.SpatialSAD_1() template_SpatialSAD('torch.FloatTensor',1,1) end
function eextest.SpatialSAD_2() template_SpatialSAD('torch.FloatTensor',1,2) end
function eextest.SpatialSAD_3() template_SpatialSAD('torch.FloatTensor',3,3) end
function eextest.SpatialSAD_4() template_SpatialSAD('torch.DoubleTensor',1,1) end
function eextest.SpatialSAD_5() template_SpatialSAD('torch.DoubleTensor',1,2) end
function eextest.SpatialSAD_6() template_SpatialSAD('torch.DoubleTensor',3,3) end

-- SpatialSAD testing:
-- TODO: add Jacobian test once derivatives are implemented
-- TODO: add tests for batch version once implemented
-- TODO: test boundary conditions

local function template_SpatialMaxMap(mtype)
   local default_type = torch.getdefaulttensortype()
   torch.setdefaulttensortype(mtype)
   local conMatrix = torch.Tensor{{1,1},{2,1},{3,2},{4,2}}
   local module = nn.SpatialMaxMap(conMatrix)
   local input = torch.Tensor{ {{0,2},{3,0}}, {{1,0},{0,4}}, {{5,0},{0,8}}, {{0,6},{7,0}} }
   local m_output = module:forward(input)
   local t_output = torch.Tensor{ {{1,2},{3,4}}, {{5,6},{7,8}} }
   mytester:assertTensorEq(m_output,t_output,precision,' - output err (type: ' .. mtype .. ')')
   torch.setdefaulttensortype(default_type)
end
function eextest.SpatialMaxMap_1() template_SpatialMaxMap('torch.FloatTensor') end
function eextest.SpatialMaxMap_2() template_SpatialMaxMap('torch.DoubleTensor') end

-- SpatialMaxMap testing:
-- TODO: add Jacobian test once derivatives are implemented
-- TODO: ?add tests for batch version once implemented
-- TODO: test boundary conditions


function eex.test_modules(tests)
   xlua.require('image',true)
   mytester = torch.Tester()
   mytester:add(eextest)
   torch.manualSeed(os.time())
   mytester:run(tests)
end
