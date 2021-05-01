from datetime import datetime as DT_, timedelta as TD_
from iqoptionapi.stable_api import IQ_Option as iq
import numpy, pandas, matplotlib.pyplot, finnhub
from getpass import getpass

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
    def download_iq(symbol: str, frame: str, rows: int = 1000, now: DT_ = None):
        """
        Download data from IQ Option database.
        Inputs:     * (str) symbol: financial instrument quote. Check IQ Option portfolio.
                    * (str) frame: timeframe label, as per trading standards (e.g.: "H1", etc).
                    * (int) rows: amount of candles to be downloaded.
                    * (datetime) now: timestamp of the last candle, from which to look backwards.
                      ...default is the timestamp representing actual date and time.
        Outputs:    * (dataframe) Data: downloaded Pandas' dataframe in (T)OHLCV format.
        ... E.g.:   - download_iq("EURUSD", "M1", 1200, datetime.datetime(2020, 1, 1))
                    - download_iq("EURUSD", "M1", 1200, datetime.datetime.now())
        """
        if not isinstance(symbol, str): raise TypeError("Symbol must be a string.")
        if not isinstance(frame, str): raise TypeError("Timeframe must be a string.")
        if not isinstance(rows, int) or (rows < 1):
            raise TypeError("Row number must be a positive integer.")
        API = iq("gsolaril@alu.itba.edu.ar", getpass("Password: ")) ; API.connect()
        if (now == None): now = DT_.now()
        if not isinstance(rows, DT_):
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
    def download_fh(symbol: str, frame: str, rows: int = 1000, now: DT_ = None):
        """
        Download data from Finnhub API database.
        Inputs:     * (str) symbol: financial instrument quote. Check IQ Option portfolio.
                    * (str) frame: timeframe label, as per trading standards (e.g.: "H1", etc).
                    * (int) rows: amount of candles to be downloaded.
                    * (datetime) now: timestamp of the last candle, from which to look backwards.
                      ...default is the timestamp representing actual date and time.
        Outputs:    * (dataframe) Data: downloaded Pandas' dataframe in (T)OHLCV format.
        ... E.g.:   - download_fh("EURUSD", "M1", 1200, datetime.datetime(2020, 01, 01))
                    - download_fh("EURUSD", "M1", 1200, datetime.datetime.now())
        """
        FCL = finnhub.Client(api_key = "c1cjh0748v6vbcpf2rlg")
        if (now == None): now = DT_.now()
        frame = MkData.frame2secs(frame)
        t = int(now.timestamp())
        t0 = t - rows*frame
        Data = FCL.forex_candles(symbol, frame//60, t0, t)
        Data.pop("s")  ;  Data = pandas.DataFrame(Data)
        Data.set_index(keys = "t", drop = True, inplace = True)
        Data.index = pandas.to_datetime(Data.index, unit = "s", utc = True)
        Data.index = Data.index.tz_convert("America/Argentina/Buenos_Aires")
        Data.columns = [_.upper() for _ in Data.columns]
        return Data.sort_index().drop_duplicates()

class Candle:

    f = matplotlib.figure.Figure
    a = matplotlib.figure.Axes

    @staticmethod
    def plot(O, H, L, C, V = None, T = None) -> (f, a):
        """
        Plot a candlestick chart with given price arrays.
        Inputs:     * (any iterable of floats) O: array with [O]pen prices.
                    * (any iterable of floats) H: array with [H]igh prices.
                    * (any iterable of floats) L: array with [L]ow prices.
                    * (any iterable of floats) C: array with [C]lose prices.
                    * (optional, any iterable of ints) V: array with [V]olume prices.
                    * (optional, any iterable of datetimes) t: array with [T]imestamps.
        Outputs:    * (matplotlib figure) fig: Figure container with default properties.
                    * (matplotlib subplot axes) ax: Axes with complete candlestick chart.
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
        n = len(O)
        fig, ax = matplotlib.pyplot.subplots()
        bu, be = (O <= C).values, (C < O).values
        x, cw = numpy.arange(n), 1/n**(1/2)
        c_back = numpy.array(fig.get_facecolor())
        c_fore = 1 - c_back   ;   c_fore[-1] = 1
        ax.bar(x[bu], H[bu] - L[bu], 1*cw, L[bu], fc = c_fore, ec = c_fore, lw = 1)
        ax.bar(x[bu], C[bu] - O[bu], 5*cw, O[bu], fc = c_fore, ec = c_fore, lw = 1)
        ax.bar(x[be], H[be] - L[be], 1*cw, L[be], fc = c_fore, ec = c_fore, lw = 1)
        ax.bar(x[be], O[be] - C[be], 5*cw, C[be], fc = c_back, ec = c_fore, lw = 1)
        if cond_V:
            ax.twinx().bar(x, V, 3*cw, fc = c_fore, ec = c_fore, lw = 1)
            ax.twinx().set_ylim(ymin = 0, ymax = 4*V.max())
            ymin, ymax = ax.get_ylim() ; yd = ymax - ymin
            ax.set_ylim(ymax = ymax, ymin = ymin - yd/3)
        if cond_T:
            try: ts = [t.strftime(format = "%y/%m/%d %H:%M") for t in T]
            except: raise TypeError("Time labels must be of datetime type.")
            ax.set_xticks(ticks = range(0, len(ts), len(ts)//30))
            ax.set_xticklabels(ts[: : len(ts)//30], rotation = 90)
        return fig, ax

if __name__ == "__main__":

    #symbol, frame, rows = "OANDA:EUR_USD", "H1", 100
    #df = MkData.download_fh(symbol, frame, rows)
    #f, a = Candle.plot(df["O"], df["H"], df["L"], df["C"], V = df["V"], T = df.index)
    #a.set_title(f"{symbol}, {frame}, {rows} rows")
    #f.savefig(fname = "testfig.jpg")
    help(MkData.download_fh)