import datetime
from typing import Final
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

date_format: Final[str] = "%m/%d/%y"

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

# samples: measurements for MACD, newest first
# periods: the length of the EMA to calculate
# returns: tuple of lists of computed MACD and signal, oldest first
def macd_signal_gen(samples: list[float], macd_period_one: int, macd_period_two: int, signal_period: int) \
    -> tuple[list[float], list[float]]:
    # how many samples are in data
    sample_count: Final[int] = len(samples)
    # first possible sample to calculate macd (last by date)
    oldest_sample = sample_count - max(macd_period_one + 1, macd_period_two + 1)

    # calculate the MACD
    left_emas = ema_calculate(macd_period_one, samples, oldest_sample)
    right_emas = ema_calculate(macd_period_two, samples, oldest_sample)
    macd = [(l - r) for (l,r) in zip(left_emas,right_emas)]

    # calculate the signal line
    first_day_signal = len(macd) - (signal_period + 1)
    signal = ema_calculate(signal_period, list(reversed(macd)),first_day_signal)

    return (macd, signal)

def main():
    period1 = 12
    perdio2 = 26
    period3 = 9

    # read the prices from file
    data = pd.read_csv("HistoricalPrices.csv", skipinitialspace=True)
    dates = list(data["Date"])
    prices = list(data["Open"])

    # generate the MACD and signal
    macd,signal = macd_signal_gen(prices, period1, perdio2, period3)

    # how many days are in data
    sample_count: Final[int] = len(prices)
    # first possible day to calculate macd (last by date)
    oldest_day = sample_count - max(period1 + 1, perdio2 + 1)
    # convert dates to datetime
    dates = [datetime.datetime.strptime(d, date_format) for d in dates]
    # cut off the dates for which we can't calculate MACD, reverse them to be in right order
    dates = dates[:(oldest_day+1)]
    dates.reverse()

    # display the plots
    fig, ax = plt.subplots()
    ax.plot_date(dates, macd, '-')
    ax.plot_date(dates[period3:], signal, '-')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(dates[0] - datetime.timedelta(days=5), dates[-1] + datetime.timedelta(days=5))
    fig.autofmt_xdate()
    fig.show()

    input()
    
if __name__ == "__main__":
    main()
    