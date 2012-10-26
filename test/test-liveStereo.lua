-- This script compute the correlation between two live images from two cameras using SAD atm
-- It calls a C function
-- Created: September 2012

-- required packages

require 'image'
require 'eex'

-- options
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SAD stereo algorith (dense match my C youtine) live demo')
cmd:text()
cmd:text('Options:')
-- global:
--cmd:option('-size', 'big', 'Input the size {big|75|mid|25|Sabine|Table|Computer}')
cmd:option('-dMax', '16', 'Maximum disparity in X-direction')
cmd:option('-dMin', '0', 'Minimum disparity in X-direction')
cmd:text()
opt = cmd:parse(arg or {})

-- initialising cameras and required needed packages

require 'xlua'
require 'camera'
width = 400
height = 300
fps = 30 
dir = 'test_vid_two'

sys.execute(string.format('mkdir -p %s',dir))

camera1 = image.Camera{idx=1,width=width,height=height,fps=fps}
camera2 = image.Camera{idx=2,width=width,height=height,fps=fps}

-- loading images

--iRc = image.loadPNG('im/imL-' .. opt.size .. '.png')
--iLc = image.loadPNG('im/imR-' .. opt.size .. '.png')

iLc = camera1:forward() -- acquiring image from the LEFT camera
iRc = camera2:forward() -- acquiring image from the RIGHT camera

-- converting in B&W

iR = image.rgb2y(iRc):float()
iL = image.rgb2y(iLc):float()

-- useful parameters

corrWindowSize = 9  -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
dMin = opt.dMin      -- Minimum Disparity in X-direction (dMin < dMax)
dMax = opt.dMax      -- Maximum Disparity in X-direction (dMax > dMin)
method = 'SAD'       -- Method used for calculating the correlation scores (SAD is the only available atm)

nr = (#iR)[2]        -- Number of row
nc = (#iR)[3]        -- Number of column

dispMap = torch.zeros(nr-(corrWindowSize-1), nc-(corrWindowSize+dMax-1)):float()  -- output Disparity Map

-- Timer

--time = sys.clock()

-- Edge detection

--require 'edgeDetector'
--dispMap = edgeDetector(iR:double()) 

-- calling the stereoC.lua routine

eex.stereo(dispMap, iR[1], iL[1], corrWindowSize, dMin, dMax)

-- printing the time elapsed

--time = sys.clock() - time
--print('dMax = ' .. dMax .. ', time elapsed = ' .. time .. 's')
--print('fps = ' .. 1/time)

-- displaying input images and Disparity Map

--image.display{image = iRc, legend = 'Image 1'}
--image.display{image = iLc, legend = 'Image 2'}
win = image.display{win = win, image = dispMap, legend = 'Disparity map dense, dMax = ' .. dMax, zoom = 3}
--end

i, totTime = 0, 0

while true do
   i = i + 1
   time = sys.clock()
   iLc = camera1:forward()
   iRc = camera2:forward()
   iR = image.rgb2y(iRc):float()
   iL = image.rgb2y(iLc):float()
   eex.stereo(dispMap, iR[1], iL[1], corrWindowSize, dMin, dMax)
   --aaa = image.convolve(dispMap:double(),image.gaussian(5))
   time = sys.clock() - time
   totTime = totTime + time
   if i == 10 then
      print('fps = ' .. 10/totTime)
      i, totTime = 0, 0
   end
   image.display{win = win, image = dispMap, legend = 'Disparity map dense, dMax = ' .. dMax, zoom = 3}
end

