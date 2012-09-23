local SpatialSAD, parent = torch.class('nn.SpatialSAD', 'nn.Module')

function SpatialSAD:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)

   self:reset()
end

function SpatialSAD:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   self.weight:apply(function()
                        return torch.uniform(-stdv, stdv)
                     end)
   self.bias:apply(function()
                      return torch.uniform(-stdv, stdv)
                   end)   
end

function SpatialSAD:templates(templates)
   if templates then
      if templates:size(1) ~= self.nOutputPlane then
         error('templates() you must provide <nOutputPlane> templates')
      end
      self.weight:copy(templates)
      return self
   end
   return self.weight
end

function SpatialSAD:updateOutput(input)
   return input.nn.SpatialSAD_updateOutput(self, input)
end

function SpatialSAD:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialSAD_updateGradInput(self, input, gradOutput)
   end
end

function SpatialSAD:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialSAD_accGradParameters(self, input, gradOutput, scale)
end
