local SpatialMaxMap, parent = torch.class('nn.SpatialMaxMap', 'nn.Module')

-- TODO: implement backprop - connTableRev and indices
function SpatialMaxMap:__init(conMatrix)
   parent.__init(self)

   self.connTable = conMatrix
   --self.connTableRev = constructTableRev(conMatrix)
   self.nInputPlane = self.connTable:select(2,1):max()
   self.nOutputPlane = self.connTable:select(2,2):max()

   --self.indices = torch.Tensor()
end

function SpatialMaxMap:updateOutput(input)
   input.nn.SpatialMaxMap_updateOutput(self, input)
   return self.output
end

function SpatialMaxMap:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxMap_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialMaxMap:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   --self.indices:resize()
   --self.indices:storage():resize(0)
end
