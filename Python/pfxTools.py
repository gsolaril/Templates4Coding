import time, numpy, pandas, matplotlib.pyplot, finnhub, sys, itertools
from datetime import datetime as DT_, timedelta as TD_
from iqoptionapi.stable_api import IQ_Option as iq
from getpass import getpass

class MathFx:

    @staticmethod
    def flog(n: float, b: float = 10):
        """
        Fake logarithm. Divides the logarithm base "b" of argument "n" into its non-decimal and
        decimal part. Keeps its whole part and linearly maps the remainder into the final result,
        in the interval between the whole part and the next integer. Method described in example
        below.
        Inputs:     * (float) "n": argument. Must be real and positive.
                    * (float) "b": base. Must be real and positive.
        Output:     * (float) Output.
        Examples:   - flog(n, 10), where "n" is any integer between 0 and 1000...
                      n => 1, 2, 3,..., 10, 20, 30, ..., 100, 200, 300, ..., 1000,
                      f => 0, 0.1*, 0.2*,..., 1, 1.1*, 1.2*, ..., 2, 2.1*, 2.2*, ..., 3
                      Asterisk "*" denotes that the decimal part is periodic.
                    - flog(47, 10)... "n = 47", "b = 10".
                        Step 1: Keep the whole part "W" of its normal log.
                        ... "47 is between 10 (10^1) and 100 (10^2)" ==> "W = 1"
                        Step 2: Linearly map decimal remainder between "[10ᵂ, 10ᵂ⁺¹]".
                        ... "47 is 37 apart from 10, 100 is 90 apart from 10" ==> D = 37/90
                        Step 3: Add both parts together, and return.
                        ... "f = W + D = 1 + 37/90" ==> return "f = 1.41*"
        """
        W = int(numpy.log(n)/numpy.log(b))
        D = (n - W)/((b - 1)*(b**W))
        return W + D

class MkData:

    mults = {"S": 1, "M": 60, "H": 60, "D": 24, "W": 7}

    @staticmethod
    def secs2frame(secs: int) -> str:
        """
        Converts a numeric time step into the standard trading timeframe label.
        Inputs:     * (int) secs: time step measured in seconds.
        Outputs:    * (str) frame: timeframe label [time-unit + amount].
        ...E.g.:    - secs2frame(1) = "S1"
                    - secs2frame(120) = "M2"
                    - secs2frame(21600) = "H6"
        """
        if not (isinstance(secs, int) and secs > 0):
            raise TypeError("Number of seconds must be a positive integer.")
        frame = ""
        for unit, qty in MkData.mults.items():
            if secs/qty < 1: return frame
            secs = secs/qty
            frame = unit + str(int(secs))
        return frame

    @staticmethod
    def frame2secs(frame: str) -> int:
        """
        Converts a standard trading timeframe label into a numeric time step.
        Inputs:     * (str) frame: timeframe label [time-unit + amount].
        Outputs:    * (int) secs: time step measured in seconds.
        ...E.g.:    - frame2secs("S15") = 15
                    - frame2secs("M30") = 1800
                    - frame2secs("D1") = 86400
        """
        if not (isinstance(frame, str) and len(frame) > 1):
            raise TypeError("Timeframe must be a string with at least 2 characters.")
        unit, number, units = frame[:1], frame[1:], MkData.mults.keys()
        if not (unit in units):
            raise ValueError(f"Timeframe unit must be one of the following: {units}")
        if not number.isdigit():
            raise ValueError(f"Timeframe number must be a positive integer")
        mult, number = 1, int(number)
        if (unit == "M"): mult *= 60
        if (unit == "H"): mult *= 60*60
        if (unit == "D"): mult *= 60*60*24
        return mult*number

    @staticmethod
    def __download_iq(API, symbol: str, frame: str, rows: int = 1000, now: DT_ = None):
        """ [Private function. Use public, without trailing underscores] """
        if not isinstance(symbol, str): raise TypeError("Symbol must be a string.")
        if not isinstance(frame, str): raise TypeError("Timeframe must be a string.")
        if not isinstance(rows, int) or (rows < 1):
            raise TypeError("Row number must be a positive integer.")
        if not isinstance(now, DT_):
            raise TypeError("Reference timestamp must be of datetime datatype.")
        frame = MkData.frame2secs(frame)
        Data = pandas.DataFrame()
        t = int(now.timestamp())
        while (rows):
            n_rows = min(1000, rows)
            data = API.get_candles(symbol, frame, n_rows, t)
            data = pandas.DataFrame(data).rename(columns = {"volume": "V",
                "from": "t", "open": "O", "max": "H", "min": "L", "close": "C"})
            data.set_index(keys = "t", drop = True, inplace = True)
            data.index = pandas.to_datetime(data.index, unit = "s", utc = True)
            data.index = data.index.tz_convert("America/Argentina/Buenos_Aires")
            data.drop(columns = ["id", "at", "to"], inplace = True)
            data = data.iloc[:, [0, 3, 2, 1, 4]]
            Data = pandas.concat((data, Data))
            rows, t = rows - n_rows, t - n_rows*frame
        return Data.sort_index().drop_duplicates()

    @staticmethod
    def __download_fh(API, symbol: str, frame: str, rows: int = 1000, now: DT_ = None):
        """ [Private function. Use public, without trailing underscores] """
        if (now == None): now = DT_.now()
        frame = MkData.frame2secs(frame)
        t = int(now.timestamp())
        t0 = t - rows*frame
        Data = API.forex_candles(symbol, frame//60, t0, t)
        Data.pop("s")  ;  Data = pandas.DataFrame(Data)
        Data.set_index(keys = "t", drop = True, inplace = True)
        Data.index = pandas.to_datetime(Data.index, unit = "s", utc = True)
        Data.index = Data.index.tz_convert("America/Argentina/Buenos_Aires")
        Data.columns = [_.upper() for _ in Data.columns]
        return Data.sort_index().drop_duplicates()

    @staticmethod
    def download_iq(symbols, frame: str, rows: int = 1000,
        now: DT_ = None, email: str = "", pword: str = ""):
        """
        Download data from IQ Option database.
        Inputs:     * (any iterable) symbols: strings with financial quotes. Check IQ database.
                    * (str) frame: timeframe label, as per trading standards (e.g.: "H1", etc).
                    * (int) rows: amount of candles to be downloaded.
                    * (datetime) now: timestamp of the last candle, from which to look backwards.
                      ...default is the timestamp representing actual date and time.
                    * (str, str) Username (email) and password (pword) of IQ Option account.
                      ...if not specified, it will be asked in entry (input).
        Outputs:    * (dataframe) Data: downloaded Pandas' dataframe in (T)OHLCV format.
        ... E.g.:   - download_iq(["EURUSD"], "M1", 1200, datetime.datetime(2020, 1, 1))
                    - download_iq(["MSFT", "AAPL"], "M1", 1200, datetime.datetime.now())
        """
        try: symbols[0]
        except: raise TypeError("Array of symbols must be a non-empty iterable.")
        columns = pandas.MultiIndex.from_product((symbols, []))
        Data = pandas.DataFrame(columns = columns)
        while not (isinstance(email, str) and (email != "")): email = input("User email: ")
        while not (isinstance(pword, str) and (pword != "")): pword = getpass("Password: ")
        API = iq(email = email, password = pword) ; API.connect()
        if (now == None): now = DT_.now()
        for symbol in symbols:
            data = MkData.__download_iq(API, symbol, frame, rows, now)
            _ = pandas.MultiIndex.from_product([[symbol], data.columns])
            Data[_] = data
        return Data
    
    @staticmethod
    def download_fh(symbols, frame: str, rows: int = 1000, now: DT_ = None, key: str = ""):
        """
        Download data from Finnhub API database.
        Inputs:     * (str) symbol: financial instrument quote. Check IQ Option portfolio.
                    * (str) frame: timeframe label, as per trading standards (e.g.: "H1", etc).
                    * (int) rows: amount of candles to be downloaded.
                    * (datetime) now: timestamp of the last candle, from which to look backwards.
                      ...default is the timestamp representing actual date and time.
                    * (str) key: API key for Finnhub client connection.
                      ...if not specified, it will be asked in entry (input).
        Outputs:    * (dataframe) Data: downloaded Pandas' dataframe in (T)OHLCV format.
        ... E.g.:   - download_fh(["EURUSD"], "M1", 1200, datetime.datetime(2020, 1, 1))
                    - download_fh(["MSFT", "AAPL"], "M1", 1200, datetime.datetime.now())
        """
        try: symbols[0]
        except: raise TypeError("Array of symbols must be a non-empty iterable.")
        columns = pandas.MultiIndex.from_product((symbols, []))
        Data = pandas.DataFrame(columns = columns)
        while not (isinstance(key, str) and (key != "")): key = input("Finnhub API key: ")
        API = finnhub.Client(api_key = key)
        if (now == None): now = DT_.now()
        for symbol in symbols:
            data = MkData.__download_fh(API, symbol, frame, rows, now)
            _ = pandas.MultiIndex.from_product([[symbol], data.columns])
            Data[_] = data
        return Data
        
class Candle:

    _f = matplotlib.figure.Figure
    _a = matplotlib.figure.Axes
    _r = matplotlib.patches.Rectangle

    @staticmethod
    def plot(O, H, L, C, V = None, T = None) -> (_f, _a):
        """
        Plot a candlestick chart with given price arrays.
        Inputs:     * (any iterable of floats) O: array with [O]pen prices.
                    * (any iterable of floats) H: array with [H]igh prices.
                    * (any iterable of floats) L: array with [L]ow prices.
                    * (any iterable of floats) C: array with [C]lose prices.
                    * (optional, any iterable of ints) V: array with [V]olume prices.
                    * (optional, any iterable of datetimes) t: array with [T]imestamps.
        Outputs:    * (matplotlib figure) fig: Figure container with default properties.
                    * (matplotlib subplot/axes) ax: Axes with complete candlestick chart.
        """
        typeError = "Input data must be iterables."
        valuError = "Input data must be all of the same size."
        cond_V = not isinstance(V, type(None))
        cond_T = not isinstance(T, type(None))
        try:
            o, h, l, c = O[0], H[0], L[0], C[0]
            if cond_V: v = V[0]
            if cond_T: t = T[0]
        except: raise TypeError(typeError)
        if not (len(O) == len(H) == len(L) == len(C)): raise ValueError(valuError)
        if cond_V and not (len(O) == len(V)): raise ValueError(valuError)
        if cond_T and not (len(O) == len(T)): raise ValueError(valuError)
        N, bull, bear = len(O), list(), list()
        for n in range(N): [ bull.append(O[n] <= C[n]) , bear.append(O[n] > C[n]) ]
        fig, ax = matplotlib.pyplot.subplots()
        x, cw = numpy.arange(N), 1/N**(1/2)
        c_back = numpy.array(fig.get_facecolor())
        c_fore = 1 - c_back   ;   c_fore[-1] = 1
        ax.bar(x[bull], H[bull] - L[bull], 1*cw, L[bull], fc = c_fore, ec = c_fore, lw = 1)
        ax.bar(x[bull], C[bull] - O[bull], 5*cw, O[bull], fc = c_fore, ec = c_fore, lw = 1)
        ax.bar(x[bear], H[bear] - L[bear], 1*cw, L[bear], fc = c_fore, ec = c_fore, lw = 1)
        ax.bar(x[bear], O[bear] - C[bear], 5*cw, C[bear], fc = c_back, ec = c_fore, lw = 1)
        if cond_V:
            ax2 = ax.twinx()  ;  s = 0.1
            ax2.grid(False, axis = "y")
            v_max = 10*10**int(numpy.log10(max(V)))
            ax2.set_yticks(range(0, v_max//2, v_max//10))
            ymin, ymax = ax.get_ylim() ; yd = ymax - ymin
            ax2.bar(x, V, 3*cw, fc = c_fore, ec = c_fore)
            ax2.set_ylim(ymin = 0, ymax = (4 + s)*V.max())
            ax.set_ylim(ymax = ymax, ymin = ymin - yd/(3 - s))
        if cond_T:
            try: ts = [t.strftime(format = "%Y/%m/%d %H:%M") for t in T]
            except: raise TypeError("Time labels must be of datetime type.")
            ax.set_xticks(ticks = range(0, len(ts), len(ts)//30))
            ax.set_xticklabels(ts[: : len(ts)//30], rotation = 90)
        return fig, ax

    @staticmethod
    def plotb(X, ax: _a = None, ratio: float = 1/2) -> _a:
        """
        Plot a secondary oscillator chart below primary candlestick chart.
        Inputs:     * (any iterable of floats) X: array with oscillator values.
                    * (matplotlib subplot/axes) ax: Axes with candlestick chart.
                    * (float) ratio "a:b" where "b" would be the primary chart size. 
        Outputs:    * (matplotlib subplot/axes) a2: Axes with secondary chart.
        """
        if isinstance(ax, type(None)): ax = matplotlib.pyplot.gca()
        try: pos = ax.get_position()
        except: raise TypeError("Invalid instance of axes (\"ax\").")
        if not (0 < ratio <= 1):
            raise ValueError("Axes size ratio must be between 0 and 1.")
        try:
            candles = [x for x in ax.get_children() if isinstance(x, Candle._r)]
            if (len(X) != len(candles)//2): raise TypeError
        except: raise TypeError("Input data length must be the same as the axes.")
        x_a, dx_a = pos.x0, pos.x1 - pos.x0
        y_b, dy_a = pos.y0, pos.y1 - pos.y0
        dy_a, dy_b = dy_a/(1 + ratio), dy_a/(1 + 1/ratio)
        y_a = y_b + dy_b
        ax.set_position(pos = [x_a, y_a, dx_a, dy_a])
        a2 = ax.figure.add_axes((x_a, y_b, dx_a, dy_b), sharex = ax)
        a2.set_xticks(list(ax.get_xticks()))
        a2.set_xticklabels(list(ax.get_xticklabels()), rotation = 90)
        return a2

class ProgBar:
    def __init__(self, length: int, width: int = 50):
        """
        Create a small progress bar for tracking processes.
        Inputs:     * (int) length: amount of steps that the cycle requires.
                    * (int) width: amount of blocks that the bar will draw at 100%
        """
        self.__length, self.__width = length, width
        sys.stdout.write("\r[%s] 0%%" % (" "*width))
        self.__prog = 0
    def show(self, step: int = 1):
        """
        Increase progress of bar instance. Might add blocks depending on step size.
        Inputs:     * (int) step: how many steps has the process moved.
        """
        self.__prog = min(self.__prog + step, self.__length)
        width, track = self.__width, self.__prog/self.__length
        bar = "■"*int(width*track) + " "*int(width*(1 - track))
        sys.stdout.write("\r[%s] %d%%" % (bar, 100*track))
        sys.stdout.flush()

class Backtest:
    _cActive = ["OT", "OP", "Order", "Lot", "Sym", "SL", "TP", "WP", "Born", "fC", "fT"]
    _cClosed = ["OT", "OP", "Order", "Lot", "Sym", "CT", "CP", "WP", "Born", "Died", "Diff"]

    @staticmethod
    def binary(Strategy, Data: pandas.DataFrame) -> pandas.DataFrame: pass
    
    @staticmethod
    def trades(Strategy, Data: pandas.DataFrame, max_trades: int = 4) -> pandas.DataFrame:
        """
        First phase of backtest.
        Inputs:     * (Strategy) Standard strategy instance with "analyze" method.
                    * (pandas.DataFrame) Data: Market history dataset. If from
                      multiple instruments, inner column layer must be OHLCV.
                    * (int) max_trades: Max amount of simultaneous trades allowed.
        Outputs:    * (pandas.DataFrame) Spreadsheet with partial trades' data. Does
                      only focus to non-financial response of strategy, holding just
                      opening and closing time/price points, as well as sinks.
        """
        Active = pandas.DataFrame(columns = Backtest._cActive)
        Closed = pandas.DataFrame(columns = Backtest._cClosed)
        progbar = ProgBar(Data.shape[0] - Strategy.lookback())
        for n in range(Strategy.lookback(), Data.shape[0] + 1):
            rows = Data.iloc[n - Strategy.lookback() : n, :]
            t1 = rows.index[-2]   ;  t = rows.index[-1]
            r1 = rows.loc[t1, :]  ;  r = rows.loc[t, :]
            inds, sign = Strategy.analyze(rows)
            to_close, t_calc = list(), time.time()
            for label, value in inds.items():
                Data.loc[t, ("Inds", label)] = value
            Data.loc[t, "Delay"] = time.time() - t_calc
            for index in Active.index:
                trade = Active.loc[index, :].values
                t_op, p_op, order, lot, sym, p_sl, p_tp, p_wp, born, fC, fT = trade
                p_min = min(r[sym]["L"], r[sym]["O"], r1[sym]["C"])
                p_max = max(r[sym]["H"], r[sym]["O"], r1[sym]["C"])
                p_cl, died = fC(rows, inds)
                if (p_tp != None) and (p_min <= p_tp <= p_max): p_cl, died = p_tp, "TP"
                if (p_sl != None) and (p_min <= p_sl <= p_max): p_cl, died = p_sl, "SL"
                if (order > 0): Active.loc[index, "WP"] = min(p_wp, p_min)
                if (order < 0): Active.loc[index, "WP"] = max(p_wp, p_max)
                if (died == "SL"): Active.loc[index, "WP"] = p_cl
                diff = order*(p_cl - p_op)
                trade = [t_op, p_op, order, lot, sym, t, p_cl, p_wp, born, died, diff]
                if died:
                    trade = dict(zip(Backtest._cClosed, trade))
                    Closed = Closed.append(trade, ignore_index = True)
                    to_close.append(index); continue
                Active.loc[index, "SL"] = fT(rows, inds)
            if to_close: Active.drop(index = to_close, inplace = True)
            if sign["Order"] and (len(Active) < max_trades):
                Active = Active.append(sign, ignore_index = True)
            progbar.show()
        return Closed

    @staticmethod
    def project(Data: pandas.DataFrame, Trades: pandas.DataFrame, p_size: float, p_value: float,
                funds: float = 10000, i_lot: float = 1, f_comp: float = 1, leverage: int = 100):
        """
        Second phase of backtest, where monetary variables are analyzed. We chose to separate
        between "trades()" and "timeline()" so as to be able to simulate different financial
        scenarios (different initial capital, different compounding method, etc.) without
        having to repeat the backtesting process, which may take time.
        Inputs:     * (pandas.DataFrame) "Data": Market history dataframe, used in "trades()".
                    * (pandas.DataFrame) "Trades": trades' spreadsheet from first phase.
                    * (float, float) "ps" and "pv": Point size and point value, respectively.
                    * (float, float) "in_cap" and "in_lot": Starting capital and lot size.
                    * (float) "f_comp": Compounding factor. Lot after trade will always change
                      by "f_comp × R" where "R" is the return generated by the previous trade.
                    * (int) "lev": Leverage.
        Outputs:    * (pandas.DataFrame) "Account": How do the financial variables would vary
                      along the backtesting timeline: funds, equity, margin and margin level.
                      Also, "Trades" gets the following new columns:
                        - "USD": Absolute profit (monetary units) of each trade.
                        - "RET": Relative profit (%) of each trade with respect to funds.
                        - "Sink": Floating money loss at the worst price ("WP").
                        - "Margin": Margin consumed from brokerage, by each trade.
        """
        Trades["PTS"] = Trades["Diff"] / p_size
        Trades["USD"] = (Trades["Lot"] * Trades["PTS"] * p_value).round(2)
        compound_rate = (1 + f_comp*Trades["USD"]/funds).cumprod()
        compound_rate = compound_rate.shift().fillna(1)
        Trades["Lot"] = (Trades["Lot"] * compound_rate).round(2)
        Trades["USD"] = Trades["Lot"] * Trades["PTS"] * p_value
        account_funds = Trades["USD"].cumsum() + funds
        Trades["RET"] = account_funds.pct_change().fillna(0)
        Trades["Sink"] = - abs(Trades["WP"] - Trades["OP"])
        Trades["Sink"] *= Trades["Lot"] * p_value / p_size
        Trades["Margin"] = Trades["OP"] * Trades["Lot"]
        Trades["Margin"] *= p_value / (p_size * leverage)
        #################################################
        c = ["Funds", "Equity", "Margin", "Level"]
        Funds = Trades.copy()[["CT", "USD"]]
        Funds["USD"] = Funds["USD"].cumsum() + funds
        Funds = Funds.drop_duplicates(subset = "CT", keep = "last")
        Account = pandas.DataFrame(index = Data.index, columns = c)
        Account.at[Data.index[0], :] = [funds, funds, 0, 0]
        for i in Funds.index:
            Account.at[Funds["CT"][i], c[: 2]] = Funds["USD"][i]
        Account.ffill(inplace = True)
        for i in Trades.index:
            t1, t2, s, m = Trades.loc[i, ["OT", "CT", "Sink", "Margin"]]
            Account.loc[t1 : t2, ["Equity", "Margin"]] += [-s, +m]
        Account["Level"] = Account["Equity"] / Account["Margin"]
        return Account

    @staticmethod
    def stats(Trades: pandas.DataFrame) -> pandas.DataFrame:
        """
        Third phase of backtest, where results are statistically analyzed.
        Inputs:     * (pandas.DataFrame) "Trades": trades' spreadsheet from previous phases.
        Outputs:    * (pandas.DataFrame) "Stats": spreadsheet holding various statistical
                        results and metrics.
                        - When second "project()" phase has been included, analysis is made on
                        3 distinct "scores": points ("PTS"), money ("USD") and returns ("RET").
                        - When second phase is omitted, calculations are only made on price
                        difference outcomes ("Diff") so only one column is returned. Results
                        should be indeed equal to "PTS", albeit divided by point size.
        """
        from scipy.stats import norm
        nt = Trades.shape[0] ; columns = ["PTS", "USD", "RET"]
        if (Trades.shape[1] != 16): columns = ["Diff"]
        Stats = pandas.DataFrame(columns = columns)
        Stats.loc["Num :)", :] = (Trades["Diff"] > 0).sum()
        Stats.loc["Num :(", :] = (Trades["Diff"] < 0).sum()
        Stats.loc["R. Hit", :] = Stats.loc["Num :)", :]/nt
        dtime = (Trades["CT"] - Trades["OT"]).values
        Stats.loc["Avg secs", :] = dtime.mean() / 1e9
        dtime = Trades["CT"].max() - Trades["OT"].min()
        Stats.loc["Trades/hr", :] = 1440*nt/dtime.seconds
        for c in columns:
            m_, s_ = Trades.loc[:, c].mean(), Trades.loc[:, c].std()
            SH_, sh_ = (m_/s_), numpy.sqrt(nt)*(m_ / s_)
            Stats.at["Diff :)", c] = (Trades.loc[Trades["Diff"] > 0, c]).sum()
            Stats.at["Max :)", c] = (Trades.loc[Trades["Diff"] > 0, c]).max()
            Stats.at["Mean :)", c] = (Trades.loc[Trades["Diff"] > 0, c]).mean()
            Stats.at["StDv :)", c] = (Trades.loc[Trades["Diff"] > 0, c]).std()
            Stats.at["Diff :(", c] = (Trades.loc[Trades["Diff"] < 0, c]).sum()
            Stats.at["Min :(", c] = (Trades.loc[Trades["Diff"] < 0, c]).min()
            Stats.at["Mean :(", c] = (Trades.loc[Trades["Diff"] < 0, c]).mean()
            Stats.at["StDv :(", c] = (Trades.loc[Trades["Diff"] < 0, c]).std()
            Stats.at["Diff", c] = (Trades.loc[:, c]).sum()
            Stats.at["Mean", c], Stats.at["StDv", c] = m_, s_
            Stats.at["Skew", c] = (Trades.loc[:, c]).skew()
            Stats.at["Sharpe", c], Stats.at["sharpe", c] = SH_, sh_
            Stats.at["Var", c] = norm.ppf(0.05) * s_ + m_
            Stats.at["var", c] = norm.ppf(0.05) * s_ + m_
            Stats.at["Z-prob", c] = norm.cdf(- SH_)
            Stats.at["z-prob", c] = norm.cdf(- sh_)
        return Stats

    @staticmethod
    def plot(): pass
        #return _f, _a

if (__name__ == "__main__"):

    account = {"email": "gsolaril@alu.itba.edu.ar", "pword": "12345678"}
    symbols, frame, rows = ["USDCAD", "GBPNZD"], "H1", 50 ## IQ and FH have different quote labels!
    dataset = MkData.download_iq(symbols, frame, rows, **account)
    print(dataset)
    #f, a = Candle.plot(df["O"], df["H"], df["L"], df["C"], V = df["V"], T = df.index)
    #a.set_title(f"{symbol}, {frame}, {rows} rows")
    #f.savefig(fname = "testfig.jpg")
