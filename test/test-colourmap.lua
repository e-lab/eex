-- Testing the imgraph package in order to colourise the stereo output

require 'imgraph'

-- parameters
dMin = 0
dMax = 16

-- generate the colourmap
map = image.colormap(dMax-dMin)

-- load stereo output greyscale image
stereoG = image.loadPNG('stereoOutput.png')

-- plotting the loaded image
image.display{image = stereoG, legend = 'Greyscale version'}

-- rescaling stereoG (loaded from file)
stereoG:mul(dMax-dMin)

-- colourise the greyscale map
stereoC = imgraph.colorize(stereoG)

image.display{image = stereoC, legend = 'Colourmapped version'}
