-- Requires ------------------------------------------------------------------

require 'sys'

-- Functions -----------------------------------------------------------------

-- LS - unix listing command
function ls(path) return sys.split(sys.ls(path),'\n') end

-- datasetPath - getting the environmental $DATASET variable
function datasetPath()
   ds = os.getenv("DATASETS")
   return ds
end
