-- Requires ------------------------------------------------------------------

require 'sys'

-- Functions -----------------------------------------------------------------

-- LS - unix listing command
function ls(path) return sys.split(sys.ls(path),'\n') end

-- datasetPath - getting the environmental $DATASET variable
function datasetPath()
   local ds
   ds = os.getenv("DATASETS")
   if not ds then
      io.write [[

******************************************************************************
   WARNING - $DATASET environment variable is not defined
******************************************************************************
   Please define an environment $DATASET variable

      $ export DATASET='datasetDirectoryOnCurrentMachine'

   and add it to your <.bashrc> configuration file in order to do not
   visualise this WARNING message again
******************************************************************************

Please, type the dataset directory path for the current machine:
]]
      ds = io.read()
   end
   return ds
end
