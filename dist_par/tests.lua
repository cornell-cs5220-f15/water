--
-- Basic tests
--
nx = tonumber(args[2]) or 200
npx = tonumber(args[3]) or 4

pond = {
  init = function(x,y) return 1, 0, 0 end,
  out = "pond.out",
  nx = nx,
  npx = npx
}

river = {
  init = function(x,y) return 1, 1, 0 end,
  out = "river.out",
  nx = nx,
  npx = npx
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
  npx = npx
}

wave = {
  init = function(x,y)
    return 1.0 + 0.2 * math.sin(math.pi * x), 1, 0
  end,
  out = "wave.out",
  frames = 100,
  nx = nx,
  npx = npx
}

simulate(_G[args[1]])
