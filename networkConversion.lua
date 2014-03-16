require 'nnx'

-- convert nn.SpatialConvolutionMM to nn.SpatialConvolution
function eex.NetConvertToDefault(net)
   for i, each in ipairs(net.modules) do
      if each.__typename == 'nn.SpatialConvolutionMM' then
         local kH = each.kH
         local kW = each.kW
         local dH = each.dH
         local dW = each.dW
         local iC = net.modules[i].nInputPlane
         local oC = net.modules[i].nOutputPlane
         local weight = net.modules[i].weight:resize(oC, iC, kH, kW)
         local bias   = net.modules[i].bias
         net.modules[i] = nn.SpatialConvolution(iC, oC, kH, kW, dH, dW)
         net.modules[i].weight = weight
         net.modules[i].bias   = bias
      end
   end
   return net:clone()
end

-- convert nn.SpatialConvolution to nn.SpatialConvolutionMM
function eex.NetConvertToMM(net)
   for i, each in ipairs(net.modules) do
      if each.__typename == 'nn.SpatialConvolution' then
         local kH = each.kH
         local kW = each.kW
         local dH = each.dH
         local dW = each.dW
         local iC = net.modules[i].nInputPlane
         local oC = net.modules[i].nOutputPlane
         local weight = net.modules[i].weight:resize(oC, iC*kH*kW)
         local bias   = net.modules[i].bias
         net.modules[i] = nn.SpatialConvolutionMM(iC, oC, kH, kW, dH, dW)
         net.modules[i].weight = weight
         net.modules[i].bias   = bias
      end
   end
   return net:clone()
end
