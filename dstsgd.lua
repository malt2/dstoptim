--[[ Implementation of disitrubed SGD by extending existing SGD module.

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)

dstsgd extension follows sgd code without making too many
changes. 

]]

require 'malt2'
require 'cutorch'

-- global dstorm state is maintained here since state calls can be unreliable (in the code to parallelized)
gdstate = {init=false, trinit=false, cuda=false, transport="gpu", iograph=IoNet.ALL, consist=0, debug=false, seg_idx=99, cb_size=5, notify_ack=0}

function transport_init (transport)
   print ('initializing dstorm..')
   --print(debug.traceback())
   local transport = transport or gdstate.transport 
   dstorm.init(transport)
   iProc = (dstorm.iProc())
   nProc = (dstorm.nProc())
   torch.manualSeed(iProc)
   gdstate.trinit = true
end

function malt_init(state, dim)
   if (gdstate.trinit == false) then transport_init (state.transport) end
   gdstate.cuda = state.cuda or true
   if not dstorm.WITH_NOTIFYACK==0 then
       state.notify_ack = Seg.SEGSYNC_NOTIFY_ACK
   else
       state.notify_ack = Seg.SEGSYNC_NONE
   end
   gdstate.init = true



   -- gdstate.notify_ack = Seg.SEGSYNC_NOTIFY_ACK
   notify_ack = gdstate.notify_ack or state.notify_ack
   local Red = { ORIG = Seg.REDUCE_AVG_RBUF,
     OBUF = Seg.REDUCE_AVG_RBUF_OBUF,
     -- NTACK uses ASYNC consistency setting, but provides guarantee stronger the SYNC.
     -- NTACK is "stronger than SYNC": wait for ALL rcvlist, reduce, and ACK
     --                                AND guarantees no mixed-version.
     -- quick MNIST test shows 13% faster and a bit more accurate
     --                  than Red.OBUF with SYNC.
     NTACK = Seg.REDUCE_AVG_RBUF_OBUF + notify_ack
   }

   gdstate.consist = Seg.FULL + Red.OBUF + notify_ack 
   state.iograph = state.iograph or gdstate.iograph 
   state.consist = state.seg or gdstate.consist 
   print (state.seg_idx)
   print (state.iograph)
   print (state.consist)

   iProc = dstorm.iProc()
   nProc = dstorm.nProc()
   gdstate.cb_size = state.cb_size or gdstate.cb_size

   if gdstate.debug then print("Ready to add_segment("..gdstate.seg_idx..", IoNet.ALL, Seg.FULL+Red.ORIG+notify_ack, dim="..dim..", \"r4\"") end

   dstorm.add_segment(gdstate.seg_idx, state.iograph, Seg.FULL + Red.OBUF + notify_ack, dim, "r4") -- fixed, now.
   print("iProc="..iProc.." : Back from add_segment")
   if iProc == 0 then
       local seg_idx = gdstate.seg_idx
       print(" for seg_idx = "..seg_idx.." ...")
       print(" getIoNet           "..dstorm.getIoNet(seg_idx))
       print(" getPolicy          "..dstorm.getPolicy(seg_idx))
       print(" getSegNum          "..dstorm.getSegNum(seg_idx))
       print(" getObuf            "..dstorm.getObuf(seg_idx))
       print(" getIbuf            "..dstorm.getIbuf(seg_idx))
       print(" getRbuf            "..dstorm.getRbuf(seg_idx))
       print(" getNbuf            "..dstorm.getNbuf(seg_idx))
       print(" getBufBytes        "..dstorm.getBufBytes(seg_idx))
       print(" getSegBytes        "..dstorm.getSegBytes(seg_idx))
       -- segfault for cuda here:
       -- print(" getMem             "..type(dstorm.getMem(seg_idx)))       -- lua sees the raw ccptr as "string"
       print(" getMem             "..tostring(dstorm.getMem(seg_idx)))       -- changed to return Uint8 version of intptr_t(sInfo.mem)
       print(" getDatacode        "..dstorm.getDatacode(seg_idx))
       print(" getDatasize        "..dstorm.getDatasize(seg_idx))
       print(" getCnt             "..dstorm.getCnt(seg_idx))
       print(" getSizeofMsgHeader "..dstorm.getSizeofMsgHeader(seg_idx))
       print(" getSeg_id          "..dstorm.getSeg_id(seg_idx))
       print(" getFmtValue        "..dstorm.getFmtValue(seg_idx))
       if dstorm.getValid(seg_idx) then
           print(" segment          is valid")
       else
           print(" segment          is invalid")
       end
   end
   dstorm:barrier()
-- if iProc==0 then print(ionet.help()) end


--mynet = IoNet.HALTON                  -- no cvgce for 8, 12 machines
-- mynet = IoNet.ALL                     -- nice cvgce
-- mynet = ionet.halton(nProc,nProc/2) -- asim fixme


end


function optim.dstsgd(opfunc, x, config, state)
   --  print ('************************DEBUG: IN optim.dstsgd**********************************************************')
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   local mom = config.momentum or 0
   local damp = config.dampening or mom
   local nesterov = config.nesterov or false
   local lrs = config.learningRates
   local wds = config.weightDecays
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter

   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- initialize

   if  ((gdstate.init==false))  then
     malt_init(state, x:size(1))
     dstorm.barrier()
   end
  
   -- (1) evaluate f(x) and df/dx
   local fx,dfdx = opfunc(x)

   -- (2) weight decay with single or individual parameters
   if wd ~= 0 then
      dfdx:add(wd, x)
   elseif wds then
      if not state.decayParameters then
         state.decayParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.decayParameters:copy(wds):cmul(x)
      dfdx:add(state.decayParameters)
   end

   -- (3) apply momentum
   if mom ~= 0 then
      if not state.dfdx then
         state.dfdx = torch.Tensor():typeAs(dfdx):resizeAs(dfdx):copy(dfdx)
      else
         state.dfdx:mul(mom):add(1-damp, dfdx)
      end
      if nesterov then
         dfdx:add(mom, state.dfdx)
      else
         dfdx = state.dfdx
      end
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)

   -- (5) parameter update with single or individual learning rates
   if lrs then
      if not state.deltaParameters then
         state.deltaParameters = torch.Tensor():typeAs(x):resizeAs(dfdx)
      end
      state.deltaParameters:copy(lrs):cmul(dfdx)
      x:add(-clr, state.deltaParameters)
   else
      --print (x:type())
      --print (dfdx:type())
      x:add(-clr, dfdx)
   end

    -- (6) do model averaging with MALT
    --

    --print ('*****iProc'.. iProc)
    --print (x)
    if (state.evalCounter%gdstate.cb_size== 0) then
      if dstorm.getTransport() == "gpu" then
        if (gdstate.debug) then print("cutorch getMemoryUsage: "..cutorch.getMemoryUsage(deviceId)) end
        local xcuda = x:cuda()                -- copy host data to gpu (temporarily)
        dstorm.store(gdstate.seg_idx, xcuda:storage())
        xcuda=nil
        if (gdstate.debug) then print("cutorch getMemoryUsage: "..cutorch.getMemoryUsage(deviceId)) end
      else
        local xfloat = x:float()
        dstorm.store(seg_idx, xfloat:storage())
        xfloat=nil
      end
      dstorm.barrier()

      --print (x:size())
      --send_bytes = d:push(seg_g, IoNetEnum.CHORD )
      send_bytes = dstorm.push(gdstate.seg_idx )
      if (gdstate.debug) then
          print("["..iProc.."] ---<store,push>---seg_idx, send_bytes ------------------ "..gdstate.seg_idx..", "..send_bytes)
      end
      dstorm.barrier()

      -- reduce 

      if (gdstate.cuda==true) and (dstorm.getTransport() == "mpi") then
        nreduce, offset, xfloat = dstorm.reduce(gdstate.seg_idx)
        x = xfloat:cuda()
      else
        nreduce, offset, x = dstorm.reduce(gdstate.seg_idx)
      end

      dstorm.barrier()

      if (gdstate.debug == true) then
          if x == nil then
              print("["..iProc.."] ---<reduce>-- nreduce = ----------------------- "..nreduce)
          else
              print("["..iProc.."] ---<reduce>-- nreduce, offset = ----------------------- "..nreduce..", "..offset..", "..type(x).." x2:type="..x:type())
              --if iProc == 0 then pretty.dump(getmetatable(t2)) end
          end
      end
      --print (x)

      dstorm.barrier()

    end

   -- (7) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end

function optim.nproc()
   if  (gdstate.trinit==false)  then
     print (sys.COLORS.red .. "WARN:Initialize malt2 before calling randperm with optim.randinit(transport). Selecting gpu as default transport \n")
     transport_init(gdstate.transport)
   end
   return nProc
end

function optim.iproc()
   if  (gdstate.trinit==false)  then
     print (sys.COLORS.red .. "WARN:Initialize malt2 before calling randperm with optim.randinit(transport). Selecting gpu as default transport \n")
     transport_init(gdstate.transport)
   end
   return iProc
end

function optim.randperm(n) 
   if  (gdstate.trinit==false)  then
     print (sys.COLORS.red .. "WARN:Initialize malt2 before calling randperm with optim.randinit(transport). Selecting gpu as default transport \n")
     transport_init(gdstate.transport)
   end
   return torch.randperm(n)
end
