--
-- Basic tests
--
nx = tonumber(args[2]) or 200
p = tonumber(args[3]) or 1
b = tonumber(args[4]) or 1

pond = {
  init = function(x,y) return 1, 0, 0 end,
  out = "pond.out",
  nx = nx,
  b = b,
  p = p
}

river = {
  init = function(x,y) return 1, 1, 0 end,
  out = "river.out",
  nx = nx,
  b = b,
  p = p
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
  b = b,
  p = p
}

wave = {
  init = function(x,y)
    return 1.0 + 0.2 * math.sin(math.pi * x), 1, 0
  end,
  out = "wave.out",
  frames = 100,
  nx = nx,
  b = b,
  p = p
}

simulate(_G[args[1]])
