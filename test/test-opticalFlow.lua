-- This script compute the correlation between two live images from two cameras using SAD atm
-- It calls a C function
-- Created: December 2012

-- required packages

require 'image'

-- c lib:
require 'eex'
require 'libimgraph'

-- options
print '==> processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SAD stereo algorith (dense match my C youtine) live demo')
cmd:text()
cmd:text('Options:')

-- global:
cmd:option('-dMax', '4', 'Maximum disparity in X-direction')
cmd:option('-dMin', '-4', 'Minimum disparity in X-direction')
cmd:option('-th', '.08', 'Background filtering [0, 2.5], 0.06 better for kSize = 3, 0.08 better for kSize = 5')
cmd:option('-kSize', '5', 'Edge kernel size {3,5}')
cmd:option('-width', '100', 'Enter the width of the camera frame (robot feasible width = 60)')
cmd:option('-UpDown', '4', 'Enter the number of preceding/succeding lines to check')
cmd:text()
opt = cmd:parse(arg or {})

-- initialising cameras and required needed packages

require 'xlua'
require 'camera'
width = tonumber(opt.width)
height = 300 * width / 400
fps = 30
--dir = 'test_vid_two'
--sys.execute(string.format('mkdir -p %s',dir))

camera1 = image.Camera{idx=0,width=width,height=height,fps=fps}
--camera2 = image.Camera{idx=2,width=width,height=height,fps=fps}

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

iCamera = camera1:forward()
--iCameraRc = camera2:forward() -- acquiring image from the RIGHT camera

-- converting in B&W

iCamera = image.rgb2y(iCamera):float()

-- useful parameters

corrWindowSize = 9  -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
dMin = opt.dMin     -- Minimum Disparity in X-direction (dMin < dMax)
dMax = opt.dMax     -- Maximum Disparity in X-direction (dMax > dMin)
UpDown = opt.UpDown -- Specifies the UpDown search

nr = (#iCamera)[2]        -- Number of row
nc = (#iCamera)[3]        -- Number of column

dispMapR = nr-(corrWindowSize+2*UpDown-1)
dispMapC = nc-(corrWindowSize+dMax-dMin-1)

dispMap = torch.zeros(dispMapR, dispMapC):float()  -- output Disparity Map
--map = torch.Tensor(dMax-dMin + 1,3):float() -- allocate the space for the colourmap + the NC (grey) layer
map = image.jetColormap(dMax-dMin + 1):float() -- generate a linearly spaced colourmap around Newton-(hue)-'s wheel
--map[{ {2,dMax-dMin+1},{} }]  = image.jetColormap(dMax-dMin):float() -- generate a Jet colourmap (I made this function u.u)
map[dMax+1]:fill(.5) -- set to a neutral grey the non-computed disparity values
colourised = torch.Tensor():typeAs(dispMap) -- allocate the space for the colourised version of the disparity map

-- Initialising variables for timing and fps printing
i, totTime = 0, 0

while true do
   i = i + 1 -- Counter for the fps printing
   time = sys.clock() -- Start timing

   -- Grabbing the two colour images
   iCameraOld = iCamera
   iCamera    = camera1:forward()

   -- Converting them into a greyscale map
   iCamera = image.rgb2y(iCamera):float()

   -- Computing the edges of the LEFT image (RIGHT camera)
   require 'edgeDetector'
   edges = edgeDetector(iCamera:double(),opt.kSize):float()[1]:abs()

   -- Computing the stereo correlation
   dispMap:fill(0)
   eex.stereo(dispMap, iCamera[1], iCameraOld[1], edges, corrWindowSize, dMin, dMax, UpDown, opt.th)
   dispMap:add(-dMin)

   -- Stopping the timer and summing up totTime
   time = sys.clock() - time
   totTime = totTime + time

   -- Every 10 frames, printing out the fps value
   if i == 10 then
      print('fps = ' .. 10/totTime)
      i, totTime = 0, 0
   end

   -- Displaying the stereo correlation map
   --win = image.display{win = win, image = dispMap, legend = 'Disparity map, dMax = ' .. dMax .. ', th = ' .. opt.th, zoom = 3*400/width}
   dispMap.imgraph.colorize(colourised, dispMap, map)
   winc = image.display{win = winc, image = colourised, legend = 'Colour disparity map, dMax = ' .. dMax .. ', th = ' .. opt.th, zoom = 3*400/width}
end