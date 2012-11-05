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
cmd:option('-th', '.06', 'Background filtering [0, 2.5]')
cmd:option('-save', 'n', 'Save to file {[n],y}?')
cmd:text()
opt = cmd:parse(arg or {})

-- loading images

iLc = image.loadPNG('im/imL-' .. opt.size .. '.png')
iRc = image.loadPNG('im/imR-' .. opt.size .. '.png')

-- converting in B&W

iL = image.rgb2y(iLc):float()
iR = image.rgb2y(iRc):float()

-- computing the edges of the LEFT image
require 'edgeDetector'
edges = edgeDetector(iL:double()):float()[1]:abs()

-- useful parameters

corrWindowSize = 9  -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
dMin = opt.dMin      -- Minimum Disparity in X-direction (dMin < dMax)
dMax = opt.dMax      -- Maximum Disparity in X-direction (dMax > dMin)
method = 'SAD'       -- Method used for calculating the correlation scores (SAD is the only available atm)

nr = (#iL)[2]        -- Number of row
nc = (#iL)[3]        -- Number of column

dispMap = torch.zeros(nr-(corrWindowSize-1), nc-(corrWindowSize+dMax-1)):float()  -- output Disparity Map

-- Timer

time = sys.clock()

-- calling the stereoC.lua routine

eex.stereo(dispMap, iL[1], iR[1], edges, corrWindowSize, dMin, dMax, opt.th)

-- printing the time elapsed

time = sys.clock() - time
print('dMax = ' .. dMax .. ', time elapsed = ' .. time .. 's')

-- displaying input images and Disparity Map

--image.display{image = iLc, legend = 'Image 1'}
--image.display{image = iRc, legend = 'Image 2'}
image.display{image = dispMap, legend = 'Disparity map dense, dMax = ' .. dMax}
--image.display{image = edges, legend = 'Edges of LEFT image'}


if opt.save == 'y' then
   io.write('Input the file name (without extension): ')
   ans = io.read()
   ans = ans .. '.png'

   -- saving the result to png file

   im = torch.Tensor(1,(#dispMap)[1],(#dispMap)[2])
   im[1] = dispMap
   im:mul(1/(dMax-dMin))
   image.savePNG(ans,im)

   io.write('"' .. ans .. '" written succesfully!\n')
end
