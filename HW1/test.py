# write a function that outputs nan
import numpy as np
def output_nan():
    x = np.nan
    return x*5

if __name__ == "__main__":
    print("Output of the function:", output_nan())