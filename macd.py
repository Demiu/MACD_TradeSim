import datetime as dt
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

def draw_graphs(dates: list[dt.datetime], macd: list[float], signal: list[float]):
    # dates that MACD covers
    macd_dates = dates[(len(dates) - len(macd)):]
    # dates that Signal covers
    signal_dates = dates[(len(dates) - len(signal)):]
    # earier of the two (always should be from MACD)
    earliest_date = min(macd_dates[0], signal_dates[0])

    fig, axs = plt.subplots(2,figsize=(6.4, 9))
    # show MACD and Signal on both charts
    for i in range(2):
        axs[i].plot_date(macd_dates, macd, '-')
        axs[i].plot_date(signal_dates, signal, '-')
        axs[i].xaxis.set_major_locator(mdates.YearLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axs[i].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[i].set_xlim(earliest_date - dt.timedelta(days=5), dates[-1] + dt.timedelta(days=5))
    
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.show()

def main():
    period1 = 12
    perdio2 = 26
    period3 = 9

    # read the prices from file
    data = pd.read_csv("HistoricalPrices.csv", skipinitialspace=True)
    dates = list(data["Date"])
    prices = list(data["Open"])

    # generate the MACD and Signal
    macd,signal = macd_signal_gen(prices, period1, perdio2, period3)

    # convert dates to datetime
    dates = [dt.datetime.strptime(d, date_format) for d in dates]
    # reverse the dates to match MACD/Signal order
    dates.reverse()

    # display the plots
    draw_graphs(dates, macd, signal)

    input()
    
if __name__ == "__main__":
    main()
    