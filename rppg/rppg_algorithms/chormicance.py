import numpy as np
import scipy.signal as signal
import math 

def CHROME_DEHAAN(RGB, LPF = 0.7, HPF = 2.5, FS = 30):

  LPF = 0.7
  HPF = 2.5
  WinSec = 1.6
  FS= 30

  FN = RGB.shape[0]
  NyquistF = 1/2*FS
  B, A = signal.cheby2(3, 5, [LPF/NyquistF, HPF/NyquistF], btype="band")

  WinL = math.ceil(WinSec*FS)
  if(WinL % 2):
      WinL = WinL+1
  NWin = math.floor((FN-WinL//2)/(WinL//2))
  S = np.zeros((NWin, 1))
  WinS = 0
  WinM = int(WinS+WinL//2)
  WinE = WinS+WinL
  totallen = (WinL//2)*(NWin+1)
  S = np.zeros(totallen)

  for i in range(NWin):
      RGBBase = np.mean(RGB[WinS:WinE, :], axis=0)
      RGBNorm = np.zeros((WinE-WinS, 3))
      for temp in range(WinS, WinE):
          RGBNorm[temp-WinS] = np.true_divide(RGB[temp], RGBBase)-1
      Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
      Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
      Xf = signal.filtfilt(B, A, Xs, axis=0)
      Yf = signal.filtfilt(B, A, Ys)

      Alpha = np.std(Xf) / np.std(Yf)
      SWin = Xf-Alpha*Yf
      SWin = np.multiply(SWin, signal.windows.hann(WinL))

      if(i == -1):
          S = SWin
      else:
          temp = SWin[:int(WinL//2)]
          S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
          S[WinM:WinE] = SWin[int(WinL//2):]
      WinS = WinM
      WinM = WinS+WinL//2
      WinE = WinS+WinL
  BVP = S
  return BVP
