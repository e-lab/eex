local SpatialSADMap, parent = torch.class('nn.SpatialSADMap', 'nn.Module')

function constructTableRev(conMatrix)
   local conMatrixL = conMatrix:type('torch.LongTensor')
   -- Construct reverse lookup connection table
   local thickness = conMatrixL:select(2,2):max()
   -- approximate fanin check
   if (#conMatrixL)[1] % thickness == 0 then 
      -- do a proper fanin check and set revTable
      local fanin = (#conMatrixL)[1] / thickness
      local revTable = torch.Tensor(thickness, fanin, 2)
      for ii=1,thickness do
	 local tempf = fanin
	 for jj=1,(#conMatrixL)[1] do
	    if conMatrixL[jj][2] == ii then
	       if tempf <= 0 then break end
	       revTable[ii][tempf][1] = conMatrixL[jj][1]
	       revTable[ii][tempf][2] = jj
	       tempf = tempf - 1
	    end
	 end
	 if tempf ~= 0 then 
	    fanin = -1
	    break
	 end
      end
      if fanin ~= -1 then
	 return revTable
      end
   end
   return {}
end

function SpatialSADMap:__init(conMatrix, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.connTable = conMatrix
   self.connTableRev = constructTableRev(conMatrix)
   self.nInputPlane = self.connTable:select(2,1):max()
   self.nOutputPlane = self.connTable:select(2,2):max()
   self.weight = torch.Tensor(self.connTable:size(1), kH, kW)
   self.bias = torch.Tensor(self.nOutputPlane)
   self.gradWeight = torch.Tensor(self.connTable:size(1), kH, kW)
   self.gradBias = torch.Tensor(self.nOutputPlane)
   
   self:reset()
end

function SpatialSADMap:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
      self.weight:apply(function()
			   return torch.uniform(-stdv, stdv)
			end)
      self.bias:apply(function()
			 return torch.uniform(-stdv, stdv)
		      end)
   else
      local ninp = torch.Tensor(self.nOutputPlane):zero()
      for i=1,self.connTable:size(1) do ninp[self.connTable[i][2]] =  ninp[self.connTable[i][2]]+1 end
      for k=1,self.connTable:size(1) do
         stdv = 1/math.sqrt(self.kW*self.kH*ninp[self.connTable[k][2]])
         self.weight:select(1,k):apply(function() return torch.uniform(-stdv,stdv) end)
      end
      for k=1,self.bias:size(1) do
         stdv = 1/math.sqrt(self.kW*self.kH*ninp[k])
         self.bias[k] = torch.uniform(-stdv,stdv)
      end
   end
end

function SpatialSADMap:updateOutput(input)
   input.nn.SpatialSADMap_updateOutput(self, input)
   return self.output
end

function SpatialSADMap:updateGradInput(input, gradOutput)
   input.nn.SpatialSADMap_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialSADMap:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialSADMap_accGradParameters(self, input, gradOutput, scale)
end

function SpatialSADMap:decayParameters(decay)
   self.weight:add(-decay, self.weight)
   self.bias:add(-decay, self.bias)
end
