-- This script compute the correlation between two images using SAD atm
-- It calls a C function
-- Created: September 2012

-- required packages

require 'image'
require 'eex'

-- options
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SAD stereo algorith (dense match my C youtine)')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-size', 'big', 'Input the size {big|75|mid|25|Sabine|Table|Computer}')
cmd:option('-dMax', '16', 'Maximum disparity in X-direction')
cmd:option('-dMin', '0', 'Minimum disparity in X-direction')
cmd:text()
opt = cmd:parse(arg or {})

-- loading images

iLc = image.loadPNG('im/imL-' .. opt.size .. '.png')
iRc = image.loadPNG('im/imR-' .. opt.size .. '.png')

-- converting in B&W

iL = image.rgb2y(iLc):float()
iR = image.rgb2y(iRc):float()

-- useful parameters

corrWindowSize = 9  -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
dMin = opt.dMin      -- Minimum Disparity in X-direction (dMin < dMax)
dMax = opt.dMax      -- Maximum Disparity in X-direction (dMax > dMin)
method = 'SAD'       -- Method used for calculating the correlation scores (SAD is the only available atm)

nr = (#iL)[2]        -- Number of row
nc = (#iL)[3]        -- Number of column

--[[ Test

for i = 5, 9 do
   dMax = 2^i ]]

dispMap = torch.zeros(nr-(corrWindowSize-1), nc-(corrWindowSize+dMax-1)):float()  -- output Disparity Map

-- Timer

time = sys.clock()

-- calling the stereoC.lua routine

eex.stereo(dispMap, iL[1], iR[1], corrWindowSize, dMin, dMax)

-- printing the time elapsed

time = sys.clock() - time
print('dMax = ' .. dMax .. ', time elapsed = ' .. time .. 's')

-- displaying input images and Disparity Map

--image.display{image = iLc, legend = 'Image 1'}
--image.display{image = iRc, legend = 'Image 2'}
image.display{image = dispMap, legend = 'Disparity map dense, dMax = ' .. dMax}
--end
