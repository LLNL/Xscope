
import numpy
import random
import math

def initialize():
  random.seed(1) 

def is_inf_pos(val):
  if math.isinf(val):
    return val > 0.0
  return False

def is_inf_neg(val):
  if math.isinf(val):
    return val < 0.0
  return False

def is_under_pos(val):
  if numpy.isfinite(val):
    if val > 0.0 and val < 2.22e-308:
      return True
  return False

def is_under_neg(val):
  if numpy.isfinite(val):
    if val < 0.0 and val > -2.22e-308:
      return True
  return False

def is_normal(val):
  return (not is_inf_pos(val)) and (not is_inf_neg(val)) and (not is_under_pos(val)) and (not is_under_neg(val)) and (not math.isnan(val))

def fp64_generate_number():
  # Generate exponent (11 bits)
  # Exponent cannot be 1 (Nan, Inf)
  number = 0.0
  while(True):
    man = random.random()
    exp = random.randrange(-1022, 1023)
    number = man * math.pow(2,exp)
    if is_normal(number):
      break

  return number

def fp64_generate_numbers(n: int):
  ret = []
  for i in range(n):
    ret.append(fp64_generate_number())
  return ret

if __name__ == '__main__':
  initialize()
  for i in range(50):
    f = fp64_generate_number()
    print(f)
