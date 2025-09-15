import numpy as np

def wrapped_angle_difference(theta1, theta2) -> float:
  diff = (theta2 - theta1 + np.pi) % np.pi * 2 - np.pi
  return diff + np.pi * 2 if diff < -np.pi else diff