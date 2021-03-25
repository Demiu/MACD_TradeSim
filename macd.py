from typing import Final
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

# 0th sample is p0 in the formula
# takes the array of terms used to calculate the previous numerator (to speed up calculations) and the newest sample
# returns: a tuple of the EMA numerator and the list of terms used to create it
def ema_numerator_next(sample: float, alpha: float, prev_array: list[float]) -> tuple[float, list[float]]:
    # remove last sample, increase the exponent for (1-alpha) for all past samples
    past_samples = [(x * (1-alpha)) for x in prev_array[:-1]]
    # add the newest sample at the beginning of the new terms
    terms = [sample] + past_samples
    value = sum(terms)
    return (value,terms)

# samples: measurements, newest first, has to be longer than first sample by N+1
# first_sample: the index of the oldest sample to calculate (10 means calculate 10 EMAs for samples 0..10)
# returns: a list of emas from 0th sample to first_sample, oldest first (reverse input order)
def ema_calculate(N: int, samples: list[float], first_sample: int) -> list[float]:
    sample_count: Final[float] = N+1
    alpha: Final[float] = ema_alpha(N)
    divisor: Final[float] = ema_denominator(N, alpha)
    # we can't sample data that would require measurement outside of provided range
    assert len(samples) >= (sample_count + first_sample)
    # sample range for the first (oldest) EMA term
    first_ema_samples: Final[list[float]] = samples[first_sample:(first_sample+sample_count)]
    numerator, terms = ema_numerator(first_ema_samples, alpha)
    # create the return variable and add the first ema
    ema_list = []
    ema_list.append(numerator / divisor)
    # iterate over the rest of indicies and calculate next EMAs ()
    for sidx in reversed(range(first_sample)):
        sample = samples[sidx]
        numerator, terms = ema_numerator_next(sample,alpha,terms)
        ema_list.append(numerator / divisor)
    
    return ema_list
    
if __name__ == "__main__":
    left_ema_N = 12
    right_ema_N = 26
    signal_ema_N = 9

    # read the prices from file
    data = pd.read_csv("HistoricalPrices.csv", skipinitialspace=True)
    dates = data["Date"]
    prices = list(data["Open"])
    # days in our data
    day_count: Final[int] = len(prices)
    first_day = day_count - max(left_ema_N + 1, right_ema_N + 1)

    # calculate the MACD
    left_emas = ema_calculate(left_ema_N, prices, first_day)
    right_emas = ema_calculate(right_ema_N, prices, first_day)
    macd_list = [(l - r) for (l,r) in zip(left_emas,right_emas)]

    # calculate the signal line
    first_day_signal = len(macd_list) - (signal_ema_N + 1)
    signal_emas = ema_calculate(signal_ema_N, list(reversed(macd_list)),first_day_signal)

    # pad the signal line at the beginning to align it with the MACD
    signal_emas = ([signal_emas[0]] * signal_ema_N) + signal_emas

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(macd_list)
    ax.plot(signal_emas)
    fig.show()

    input()
