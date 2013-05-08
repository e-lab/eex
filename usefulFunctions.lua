-- Requires ------------------------------------------------------------------
require 'sys'

-- Functions -----------------------------------------------------------------

-- eex.ls - unix listing command
function eex.ls(path) return sys.split(sys.ls(path),'\n') end

-- eex.datasetPath - getting the environmental $EEX_DATASETS variable
function eex.datasetsPath()
   local ds
   ds = os.getenv("EEX_DATASETS")
   if not ds then
      io.write [[

******************************************************************************
   WARNING - $EEX_DATASETS environment variable is not defined
******************************************************************************
   Please define an environment $EEX_DATASETS variable

      $ export EEX_DATASETS='datasetsDirectoryOnCurrentMachine'

   and add it to your <.bashrc> configuration file in order to do not
   visualise this WARNING message again
******************************************************************************

Please, type the dataset directory path for the current machine:
]]
      ds = io.read()
      io.write '\n'
   end
   return ds
end
