optim ={}
optim = require 'optim'
--dopt  = require 'dstoptim.dstsgd'
optim[#optim+1] = require 'dstoptim.dstsgd' 

return optim 
