import pandas as pd

def ema_alpha(N: int) -> float:
    return 2 / (N + 1)

def ema_denominator(N: int, alpha: float) -> float:
    return sum(map(lambda x: (1-alpha)**x, range(N+1)))

# returns the list of terms in the sum in EMA numerator
# 0th sample is p0 in the formula
def ema_numerator_array(samples: list[float], alpha: float) -> list[float]:
    exponents = range(len(samples))
    return list(map(lambda p, e: p*((1-alpha)**e), samples, exponents))

# returns a tuple of the EMA numerator and the list of terms used to create it
# 0th sample is p0 in the formula
def ema_numerator(samples: list[float], alpha: float) -> tuple(float, list[float]):
    array = ema_numerator_array(samples, alpha)
    value = sum(array)
    return (value,array)

# returns a tuple of the EMA numerator and the list of terms used to create it
# 0th sample is p0 in the formula
# takes the array of terms used to calculate the previous numerator to speed up calculations
def ema_numerator_next(sample: float, alpha: float, prev_array: list[float]) -> tuple(float, list[float]):
    new_samples = [sample] + prev_array[:-1]
    return ema_numerator(new_samples, alpha)
    
if __name__ == "__main__":
    left_ema_N = 12
    left_ema_alpha = ema_alpha(left_ema_N)
    left_ema_divisor = ema_denominator(left_ema_N, left_ema_alpha)

    right_ema_N = 26
    right_ema_alpha = ema_alpha(right_ema_N)
    right_ema_divisor = ema_denominator(right_ema_N, right_ema_alpha)

    # read the prices from file
    prices = pd.read_csv("HistoricalPrices.csv", skipinitialspace=True)
    prices = prices["Open"]

    # days in our data
    day_count = len(prices.index)
    # -1 because for N=x we need x+1 days sampled
    first_day = day_count - max(left_ema_N, right_ema_N) - 1 

    

    print(prices[first_day:])
