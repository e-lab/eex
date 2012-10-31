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

---------------------------------------------------------------------------------
-- INFO (for a correct connection of camera1 & 2)
-- camera1 is the camera connected to the USB plug closest to the DC power supply
-- camera2, instead, is the one closer to the user

-- In this program camera1 is supposed to serve as the LEFT camera,
-- whereas camera2 shall match the RIGHT camera. The LEFT and RIGHT cameras
-- provide respectively the RIGHT- and LEFT-shifted images

-- iCameraX[c]: {i}mage from {Camera} {X} [{c}olour version; greyscale otherwise]
---------------------------------------------------------------------------------

iCameraLc = camera1:forward() -- acquiring image from the LEFT camera
iCameraRc = camera2:forward() -- acquiring image from the RIGHT camera

-- converting in B&W

iCameraL = image.rgb2y(iCameraLc):float()
iCameraR = image.rgb2y(iCameraRc):float()

-- useful parameters

corrWindowSize = 9  -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
dMin = opt.dMin      -- Minimum Disparity in X-direction (dMin < dMax)
dMax = opt.dMax      -- Maximum Disparity in X-direction (dMax > dMin)
method = 'SAD'       -- Method used for calculating the correlation scores (SAD is the only available atm)

nr = (#iCameraR)[2]        -- Number of row
nc = (#iCameraR)[3]        -- Number of column

dispMap = torch.zeros(nr-(corrWindowSize-1), nc-(corrWindowSize+dMax-1)):float()  -- output Disparity Map

-- Initialising variables for timing and fps printing
i, totTime = 0, 0

while true do
   i = i + 1 -- Counter for the fps printing
   time = sys.clock() -- Start timing

   -- Grabbing the two colour images
   iCameraLc = camera1:forward()
   iCameraRc = camera2:forward()

   -- Converting them into a greyscale map
   iCameraR = image.rgb2y(iCameraRc):float()
   iCameraL = image.rgb2y(iCameraLc):float()

   -- Computing the stereo correlation
   eex.stereo(dispMap, iCameraR[1], iCameraL[1], corrWindowSize, dMin, dMax)

   -- Stopping the timer and summing up totTime
   time = sys.clock() - time
   totTime = totTime + time

   -- Every 10 frames, printing out the fps value
   if i == 10 then
      print('fps = ' .. 10/totTime)
      i, totTime = 0, 0
   end

   -- Displaying the stereo correlation map
   win = image.display{win = win, image = dispMap, legend = 'Disparity map dense, dMax = ' .. dMax, zoom = 3}
end

