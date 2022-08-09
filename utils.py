max_normal = 1e+307

def bounds(split, num_input, input_type="fp"):
  b = []
  upper_lim = max_normal
  lower_lim = -max_normal
  if input_type == "exp":
    upper_lim = 307
    lower_lim = -307
  if split == "whole":
    if num_input == 1:
      b.append({'x0': (lower_lim, upper_lim)})
    elif num_input == 2:
      b.append({'x0': (lower_lim, upper_lim), 'x1': (lower_lim, upper_lim)})
    else:
      b.append({'x0': (lower_lim, upper_lim), 'x1': (lower_lim, upper_lim), 'x2': (lower_lim, upper_lim)})
      
  elif split == "two":
    if num_input == 1:
      b.append({'x0': (lower_lim, 0)})
      b.append({'x0': (0, upper_lim)})
    elif num_input == 2:
      b.append({'x0': (lower_lim, 0), 'x1': (lower_lim, 0)})
      b.append({'x0': (0, upper_lim), 'x1': (0, upper_lim)})
    else:
      b.append({'x0': (lower_lim, 0), 'x1': (lower_lim, 0), 'x2': (lower_lim, 0)})
      b.append({'x0': (0, upper_lim), 'x1': (0, upper_lim), 'x2': (0, upper_lim)})
      
  else:
    limits = [0.0, 1e-307, 1e-100, 1e-10, 1e-1, 1e0, 1e+1, 1e+10, 1e+100, 1e+307]
    ranges = []
    if input_type == "exp":
      for i in range(len(limits)-1):
        x = limits[i]
        y = limits[i+1]
        t = (min(x,y), max(x,y))
        ranges.append(t)
    else:
      for i in range(len(limits)-1):
        x = limits[i]
        y = limits[i+1]
        t = (min(x,y), max(x,y))
        ranges.append(t)
        x = -limits[i]
        y = -limits[i+1]
        t = (min(x,y), max(x,y))
        ranges.append(t)
    if num_input == 1:
      for r1 in ranges:
        b.append({'x0': r1})
    elif num_input == 2:
      for r1 in ranges:
        for r2 in ranges:
          b.append({'x0': r1, 'x1': r2})
    else:
      for r1 in ranges:
        for r2 in ranges:
          b.append({'x0': r1, 'x1': r2, 'x2': r2})
  return b

def are_we_done(func, recent_val, exp_name):
  global found_inf_pos, found_inf_neg, found_under_pos, found_under_neg

  # Finding INF+
  if 'max_inf' in func.__name__:
    if found_inf_pos:
      return True
    else:
      if is_inf_pos(recent_val):
        found_inf_pos = True
        save_results(recent_val, exp_name)
        return True

  # Finding INF-
  elif 'min_inf' in func.__name__:
    if found_inf_neg:
      return True
    else:
      if is_inf_neg(recent_val):
        found_inf_neg = True
        save_results(recent_val, exp_name)
        return True

  # Finding Under-
  elif 'max_under' in func.__name__:
    if found_under_neg:
      return True
    else:
      if is_under_neg(recent_val):
        found_under_neg = True
        save_results(recent_val, exp_name)
        return True

  # Finding Under+
  elif 'min_under' in func.__name__:
    if found_under_pos:
      return True
    else:
      if is_under_pos(recent_val):
        found_under_pos = True
        save_results(recent_val, exp_name)
        return True

  return False
