--
-- Basic tests
--

pond = {
  init = function(x,y) return 1, 0, 0 end,
  out = "pond.out"
}

river = {
  init = function(x,y) return 1, 1, 0 end,
  out = "river.out"
}

dam = {
  init = function(x,y)
    if (x-1)*(x-1) + (y-1)*(y-1) < 0.25 then
      return 1.5, 0, 0
    else
      return 1, 0, 0
    end
  end,
  out = "dam_break.out"
}

wave = {
  init = function(x,y)
    return 1.0 + 0.2 * math.sin(math.pi * x), 1, 0
  end,
  out = "wave.out",
  frames = 100
}

simulate(_G[args[1]])
