#!/usr/bin/env torch
-- This script compute the correlation between two images using SAD atm
-- It calls a C function
-- Created: September 2012

-- required packages

require 'image'

-- c lib:
require 'eex'
require 'libimgraph'

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
cmd:option('-kSize', '5', 'Edge kernel size {3,[5]}')
cmd:option('-UpDown', '0', 'Enter the number of preceding/succeding lines to check')
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
edges = edgeDetector(iL:double(),opt.kSize):float()[1]:abs()

-- useful parameters

corrWindowSize = 9   -- Correlation Window Size, MUST BE AN ODD NUMBER!!!
dMin = opt.dMin      -- Minimum Disparity in X-direction (dMin < dMax)
dMax = opt.dMax      -- Maximum Disparity in X-direction (dMax > dMin)
UpDown = opt.UpDown -- Specifies the UpDown search

nr = (#iL)[2]        -- Number of row
nc = (#iL)[3]        -- Number of column

dispMap = torch.zeros(nr-(corrWindowSize+2*UpDown-1), nc-(corrWindowSize+dMax-1)):float()  -- output Disparity Map

-- Timer

time = sys.clock()

-- calling the stereoC.lua routine

eex.stereo(dispMap, dispMap, iL[1], iR[1], edges, corrWindowSize, dMin, dMax, UpDown, opt.th)

-- printing the time elapsed

time = sys.clock() - time
print('dMax = ' .. dMax .. ', time elapsed = ' .. time .. 's')

-- displaying input images and Disparity Map

--image.display{image = iLc, legend = 'Image 1'}
--image.display{image = iRc, legend = 'Image 2'}
image.display{image = dispMap, legend = 'Disparity map, dMax = ' .. dMax .. ', th = ' .. opt.th, zoom = 2}
--image.display{image = edges, legend = 'Edges of LEFT image'}

-- colourise the output

map = torch.Tensor(dMax-dMin + 1,3):float() -- allocate the space for the colourmap + the NC (grey) layer
--map = image.colormap(dMax-dMin + 1):float() -- generate a linearly spaced colourmap around Newton-(hue)-'s wheel
map[{ {2,dMax-dMin+1},{} }]  = image.jetColormap(dMax-dMin):float() -- generate a Jet colourmap (I made this function u.u)
map[1]:fill(.5) -- set to a neutral grey the non-computed disparity values

-- auto type
colourised = torch.Tensor():typeAs(dispMap)
dispMap.imgraph.colorize(colourised, dispMap, map)

image.display{image = colourised, legend = 'Colour disparity map, dMax = ' .. dMax .. ', th = ' .. opt.th, zoom = 2}

-- saving the result to png file

if opt.save == 'y' then
   io.write('Input the file name (without extension): ')
   ans = io.read()
   ans = ans .. '.png'
   io.write('[C]olour of [G]rey scale? ')
   type = io.read()
   if type == 'G' or type == 'g' then
      im = torch.Tensor(1,(#dispMap)[1],(#dispMap)[2])
      im[1] = dispMap
      im:mul(1/(dMax-dMin))
      image.savePNG(ans,im)
   else
      image.savePNG(ans,colourised:double())
   end
   io.write('"' .. ans .. '" written succesfully!\n')
end
