-- Requires ------------------------------------------------------------------

require 'sys'

-- Functions -----------------------------------------------------------------

-- LS - unix listing command
function ls(path) return sys.split(sys.execute('/bin/ls ' .. path),'\n') end
