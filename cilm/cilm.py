import pandas as pd
import scipy.stats as st


class CILM:
  """
  Confidence Learning Machine
  """
  def __init__(self,n,x,colA,colB,confi):

    """
    n: the length of the batch
    x: a univariate time-series (with 1 or max 2 columns if datatime is a column insted of an index).
    colNoA: column no of the numeric time-series
    colNoB: column no of the date. If date is in index then column 0
    colA: name of the column. For example FEDFUNDS/INFLATION/etc
    colB: name of the column with time series. For example DATE/YEAR/etc

    """
    self.data = x
    self.n = n
    self.colA = colA
    self.colB = colB
    self.confi = confi

  def cci(self,x):
    """
    Args:
      x:data stream

    Returns:
      pandas dataFrame with two columns Upper and Lower CI
    """
    ss = x.sum()
    n = len(x)+1
    mi = x.min()
    ma = x.max()


    A = ss - (n*ma)
    B = (ss - (n*mi))


    if(B>A):
      lower = mi
      upper = ((2*ss)-(n*mi))/(n-2)
    elif(A>B):
      upper = ((2*ss)-(n*ma))/(n-2)
      lower = ma
    else:
       upper = ma
       lower = mi
    out = pd.concat([pd.Series(lower),pd.Series(upper)],axis=1).reset_index(drop=True)

    out.columns = ["LCI","UCI"]
    return out

  def sim(self):

    data = self.data
    n = self.n
    colA = self.colA
    colB = self.colB
    confi = self.confi

    """
    Input:
      n: the length of the batch
      x: a univariate time-series (with 1 or max 2 columns if datatime is a column insted of an index).
      colNoA: column no of the numeric time-series
      colNoB: column no of the date. If date is in index then column 0
      colA: name of the column. For example FEDFUNDS/INFLATION/etc
      colB: name of the column with time series. For example DATE/YEAR/etc

    Output:

      colB: DATE/TIME
      NLCI: NAYMEN LOWER CONFIDENCE INTERVAL
      CLCI: CONFORMAL LOWER CONFIDENCE INTERVAL
      CUCI: CONFORMAL UPPER CONFIDENCE INTERVAL
      ACTUAL: ACTUUAL DATA
    """

    if type(data.index[0]) != int:

        data = pd.DataFrame(data.reset_index())

    data.columns = [colB,colA]

    au = []
    al = []
    bu = []
    bl = []
    c = []
    
    cilm = CILM(n,data,colA,colB,confi)

    
    for i in range(n, len(data)):

        x = pd.to_numeric(data[colA].iloc[i-n:i]).reset_index(drop=True)

        d = data[colB].iloc[i-n:i]

        conf = st.t.interval(confidence=confi, df=len(x)-2, loc=x.mean(), scale=st.sem(x))
        cl = cilm.cci(x[0:len(x)-2])
        au.append(cl["UCI"][0].astype(float))
        al.append(cl["LCI"][0].astype(float))
        bu.append(conf[1])
        bl.append(conf[0])
        c.append(x.iloc[n-1])

    actual = pd.Series(c)
    cilm_upper = pd.Series(au)
    cilm_lower = pd.Series(al)
    classical_upper = pd.Series(bu)
    classical_lower = pd.Series(bl)
    dat = data["DATE"].iloc[n:].reset_index(drop=True)
    out = pd.concat([dat,classical_lower,cilm_lower,classical_upper,cilm_upper,actual],axis=1)
    out.columns = [colB,"NLCI","CLCI","NUCI","CUCI","ACTUAL"]

    return out
