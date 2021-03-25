import pandas as pd
import matplotlib.pyplot as plt

def ema_alpha(N: int) -> float:
    return 2 / (N + 1)

def ema_denominator(N: int, alpha: float) -> float:
    return sum(map(lambda x: (1-alpha)**x, range(N+1)))

# returns the list of terms in the sum in EMA numerator
# 0th sample is p0 in the formula
def ema_numerator_terms(samples: list[float], alpha: float) -> list[float]:
    exponents = range(len(samples))
    return list(map(lambda p, e: p*((1-alpha)**e), samples, exponents))

# returns a tuple of the EMA numerator and the list of terms used to create it
# 0th sample is p0 in the formula
def ema_numerator(samples: list[float], alpha: float) -> tuple[float, list[float]]:
    terms = ema_numerator_terms(samples, alpha)
    value = sum(terms)
    return (value,terms)

# returns a tuple of the EMA numerator and the list of terms used to create it
# 0th sample is p0 in the formula
# takes the array of terms used to calculate the previous numerator (to speed up calculations) and the newest sample
def ema_numerator_next(sample: float, alpha: float, prev_array: list[float]) -> tuple[float, list[float]]:
    # remove last sample, increase the exponent for (1-alpha) for all past samples
    past_samples = [(x * (1-alpha)) for x in prev_array[:-1]]
    # add the newest sample at the beginning of the new terms
    terms = [sample] + past_samples
    value = sum(terms)
    return (value,terms)
    
if __name__ == "__main__":
    left_ema_N = 12
    left_ema_sample_count = left_ema_N + 1
    left_ema_alpha = ema_alpha(left_ema_N)
    left_ema_divisor = ema_denominator(left_ema_N, left_ema_alpha)

    right_ema_N = 26
    right_ema_sample_count = right_ema_N + 1
    right_ema_alpha = ema_alpha(right_ema_N)
    right_ema_divisor = ema_denominator(right_ema_N, right_ema_alpha)

    # read the prices from file
    data = pd.read_csv("HistoricalPrices.csv", skipinitialspace=True)
    dates = data["Date"]
    prices = data["Open"]

    # days in our data
    day_count = len(prices.index)
    # first possible to calculate, last by date
    first_day = day_count - max(left_ema_sample_count, right_ema_sample_count)

    left_ema_samples = prices[first_day:(first_day+left_ema_sample_count)]
    left_ema_numerator,left_ema_terms = ema_numerator(left_ema_samples, left_ema_alpha)
    left_ema = left_ema_numerator / left_ema_divisor

    right_ema_samples = prices[first_day:(first_day+right_ema_sample_count)]
    right_ema_numerator,right_ema_terms = ema_numerator(right_ema_samples, right_ema_alpha)
    right_ema = right_ema_numerator / right_ema_divisor
    
    t=max(left_ema_sample_count, right_ema_sample_count)

    macd_list=[]
    macd_list.append(left_ema - right_ema)

    # iterate from the first_day - 1 to 0th (latest) day
    for day in reversed(range(first_day)):
        sample = prices[day]

        left_ema_numerator,left_ema_terms = ema_numerator_next(sample,left_ema_alpha,left_ema_terms)
        left_ema = left_ema_numerator / left_ema_divisor

        right_ema_numerator,right_ema_terms = ema_numerator_next(sample,right_ema_alpha,right_ema_terms)
        right_ema = right_ema_numerator / right_ema_divisor

        macd_list.append(left_ema - right_ema)

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(macd_list)
    fig.show()

    input()
