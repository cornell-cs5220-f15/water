--
-- Basic tests
-- Arg 1: Test to run, Arg 2: Test dimension
nx = tonumber(args[2]) or 200
nthreads = tonumber(args[3]) or 10

pond = {
  init = function(x,y) return 1, 0, 0 end,
  out = "pond.out",
  nx = nx,
  nthreads=nthreads	
}

river = {
  init = function(x,y) return 1, 1, 0 end,
  out = "river.out",
  nx = nx,
  nthreads=nthreads		
}

dam = {
  init = function(x,y)
    if (x-1)*(x-1) + (y-1)*(y-1) < 0.25 then
      return 1.5, 0, 0
    else
      return 1, 0, 0
    end
  end,
  out = "dam_break.out",
  nx = nx,
  nthreads=nthreads	
}

wave = {
  init = function(x,y)
    return 1.0 + 0.2 * math.sin(math.pi * x), 1, 0
  end,
  out = "wave.out",
  frames = 100,
  nx = nx,
  nthreads=nthreads	
}

simulate(_G[args[1]])
