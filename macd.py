import datetime as dt
from typing import Final
from numpy import floor
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

# takes: current money, the multiplier of maximum money to spend (>=0,<=1), current held units, current price
# returns: new money and unit amounts
def buy_units(money: float, money_mul: float, units: int, price: float) -> tuple[float, int]:
    assert money_mul >= 0 and money_mul <= 1
    to_spend = money * money_mul
    bought = floor(to_spend / price)
    spent = bought * price

    return (money - spent, units + bought)

# takes: current money, current held units, the multiplier of units to sell (0<=m<=1), current price
# returns: new money and unit amounts
def sell_units(money: float, units: int, units_mul: float, price: float) -> tuple[float, int]:
    assert units_mul >= 0 and units_mul <= 1
    to_sell = floor(units * units_mul)
    gained = to_sell * price

    return (money + gained, units - to_sell) 


# computes a single, latest, trading day
# all parameters are in oldest first order
# returns: a tuple of new amount of held units and money
def trade_day(price: float, macd: list[float], signal: list[float], 
              units: int, money: float) -> tuple[int, float]:
    assert len(macd) >= 2 and len(signal) >= 2
    signal_over_macd_today: Final[bool] = signal[-1] > macd[-1]
    signal_over_macd_yday: Final[bool] = signal[-2] > macd[-2]
    macd_positive_today = macd[-1] >= 0
    macd_increased = macd[-1] > macd[-2]
    can_buy = (money / price) >= 1

    if macd_positive_today:
        if signal_over_macd_today and not signal_over_macd_yday and units > 0:
            # sell crossover, if macd is in positive (overall rising trend) 
            # sell everything
            money, units = sell_units(money, units, 1, price)
        elif can_buy:
            # we can buy at least 1 unit and we're in a raise
            # if we can do so then we crossed from negatives into positives
            # buy as much as possible
            money, units = buy_units(money, 1, units, price)
    elif can_buy:
        if macd_increased and not signal_over_macd_today:
            # macd increased and signal is under macd, so it will be raising
            money, units = buy_units(money, 0.05, units, price)

    return (units, money)

# all paramaters are in oldest first order
def trade_macd(prices: list[float], macd: list[float], signal: list[float]):
    # the relevant prices (covered by Signal)
    prices_relevant = prices[(len(prices) - len(signal)):]
    # the relevant MACDs (covered by Signal)
    macd_relevant = macd[(len(macd) - len(signal)):]

    # the starting amount
    units = 1000
    money = 0.0

    # start from the 2nd day because the trade_day looks at two days back
    for day in range(1, len(signal)):
        price = prices_relevant[day]
        macd_to_date = macd_relevant[:(day+1)]
        signal_to_date = signal[:(day+1)]
        units, money = trade_day(price, macd_to_date, signal_to_date, units, money)
    
    # last day selloff
    money += units * prices[-1]
    units = 0
    print("Final money: ", money, ", worth ", money/prices[-1], " units total")

def draw_graphs(prices: list[float], dates: list[dt.datetime], macd: list[float], signal: list[float]):
    # prices that MACD covers
    prices_relevant = prices[(len(prices) - len(macd)):]
    # dates that MACD covers
    macd_dates = dates[(len(dates) - len(macd)):]
    # dates that Signal covers
    signal_dates = dates[(len(dates) - len(signal)):]
    # earier of the two (always should be from MACD)
    earliest_date = macd_dates[0]

    fig, axs = plt.subplots(2,figsize=(6.4, 9))
    # set the shared parameters between chars
    for i in range(2):
        axs[i].xaxis.set_major_locator(mdates.YearLocator())
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axs[i].xaxis.set_minor_locator(mdates.MonthLocator())
        axs[i].set_xlim(earliest_date - dt.timedelta(days=5), dates[-1] + dt.timedelta(days=5))
    # show the MACD and Signal on the first chart
    axs[0].plot_date(macd_dates, macd, c="tab:blue", ls='-', marker='', zorder=10)
    axs[0].plot_date(signal_dates, signal, c="tab:orange", ls='-', marker='', zorder=5)
    # show the price on the second chart
    axs[1].plot_date(macd_dates, prices_relevant, c="tab:gray", ls='-', marker='', zorder=0)

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

    # reverse the prices order to match MACD/Signal, done now because macd_signal_gen expects newest first
    prices.reverse()

    trade_macd(prices, macd, signal)

    # convert dates to datetime
    dates = [dt.datetime.strptime(d, date_format) for d in dates]
    # reverse the dates to match MACD/Signal order
    dates.reverse()

    # display the plots
    draw_graphs(prices, dates, macd, signal)

    input()
    
if __name__ == "__main__":
    main()
    