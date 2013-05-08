-- Requires ------------------------------------------------------------------
require 'sys'

-- Functions -----------------------------------------------------------------

-- LS - unix listing command
function eex.ls(path) return sys.split(sys.ls(path),'\n') end

-- datasetPath - getting the environmental $DATASET variable
function eex.datasetPath()
   local ds
   ds = os.getenv("EEX_DATASETS")
   if not ds then
      io.write [[

******************************************************************************
   WARNING - $DATASET environment variable is not defined
******************************************************************************
   Please define an environment $DATASET variable

      $ export EEX_DATASETS='datasetDirectoryOnCurrentMachine'

   and add it to your <.bashrc> configuration file in order to do not
   visualise this WARNING message again
******************************************************************************

Please, type the dataset directory path for the current machine:
]]
      ds = io.read()
   end
   return ds
end
