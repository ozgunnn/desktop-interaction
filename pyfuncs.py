
import numpy as np
#from matplotlib import pyplot as plt
import pandas as pd
import scipy
import math


def main(fcm=30, fa=460, fs=500, gam_fcm=1, gam_fa=1, gam_fs=1, h=200, b=200, tw=9, tf=15, depth=400, reb_no=8, reb_d=20, axis='Strong', shape='Circular'):
    # fcm = 30
    # fa = 460
    # fs =500
    # gam_fcm = 1
    # gam_fa = 1
    # gam_fs = 1
    # h = 200
    # b = 200
    # tw = 9
    # tf = 15
    # depth =400
    # reb_no = 8
    # reb_dia = 20
    # axis = "Strong"
    # shape = 'Circular'

    NFibers = 400
    fiberDepth = depth/NFibers

    fiberCoords = np.linspace(
        0, 1, NFibers, endpoint=False)*depth+depth/NFibers/2
    df = pd.DataFrame(fiberCoords, columns=['Fiber Coordinates'])

    # distance from section face to profile for Strongs
    cProfStrong = (depth - h)/2
    # distance from section face to profile for Weaks
    cProfWeak = (depth - b)/2

    cc = 30  # clear cover
    Ecm = 1000*22*(fcm/10)**0.3
    ec1 = 0.0028  # min(0.7*fcm**0.31,2.8)/1000
    ecu1 = 0.0035
    kc = 1.05*Ecm*ec1/fcm

    Ea = 210000
    ea = fa/Ea
    fu = 1.4*fa
    esh = min(max(0.015, 0.1*fa/fu-0.055), 0.03)
    eu = 0.06
    C1 = (esh+0.25*(eu-esh))/eu
    C2 = (esh+0.4*(eu-esh))/eu
    Esh = (fu-fa)/(C2*eu-esh)
    fc1eu = fa+(C1*eu-esh)*Esh
    Es = 200000

    def shapeToRebDets(shape):
        if shape == "Rect":
            nrrow = int((reb_no+4)/4)
            rdist = (depth-2*cc-reb_d)/(nrrow-1)
            rebmat = np.zeros((nrrow, 2))
            for i in range(nrrow):  # rebar rows matrix
                if i == 0 or i == nrrow-1:
                    rebmat[i, 0] = nrrow  # %rebars per rebar row
                else:
                    rebmat[i, 0] = 2  # %rebars per rebar row
                rebmat[i, 1] = cc+reb_d/2+(i)*rdist  # %depth of rebar rows
            return rebmat, nrrow
        if shape == "Circular":
            rebmat = np.zeros((reb_no, 2))
            rebsetrad = depth/2-cc-reb_d/2
            teta = 2*math.pi/reb_no
            for k in range(reb_no):
                rebmat[k, 0] = k*teta
                rebmat[k, 1] = depth/2-rebsetrad*math.sin(teta*(k))
            return rebmat, 0

    def coordinateToRebarArea(input, shape):
        rebmat, nrrow = shapeToRebDets(shape)
        if shape == "Rect":
            for k in range(nrrow):
                if (input > rebmat[k, 1]-reb_d/2 and input < rebmat[k, 1]+reb_d/2):
                    values = rebmat[k, 0]*fiberDepth*2 * \
                        np.sqrt(abs((reb_d/2)**2-(rebmat[k, 1]-input)**2))
                    return values
                else:
                    pass
        if shape == "Circular":
            sum = 0
            for k in range(reb_no):
                if (input > rebmat[k, 1]-reb_d/2 and input < rebmat[k, 1]+reb_d/2):
                    sum += fiberDepth*2 * \
                        np.sqrt(abs((reb_d/2)**2-(rebmat[k, 1]-input)**2))
            return sum

    # %%
    def coordinateToSteelArea(input, axis):
        if axis == 'Strong':
            if (input < cProfStrong or input > depth - cProfStrong):
                return 0
            else:
                if (input < cProfStrong+tf or input > depth - cProfStrong - tf):
                    return fiberDepth*b
                else:
                    return fiberDepth*tw
        if axis == 'Weak':
            if (input < cProfWeak or input > depth - cProfWeak):
                return 0
            elif (input < cProfWeak+b/2-tw/2 or input > cProfWeak+b/2+tw/2):
                return 2*fiberDepth*tf
            else:
                return fiberDepth*h

    def coordinateToGrossArea(input, shape):
        if shape == 'Circular':
            return 2*np.sqrt((depth/2)**2-(depth/2-input)**2)*fiberDepth
        if shape == 'Rect':
            return fiberDepth*depth

    def strainToConcreteStress(input):
        if (input > 0 or input < -ecu1):
            return 0
        else:
            eta = abs(input/ec1)
            return -fcm*(kc*eta-eta**2)/(1+(kc-2)*eta)

    def strainToProfileStress(input):
        if abs(input) < ea:
            return input*Ea
        elif abs(input) < esh:
            return input/abs(input)*fa
        elif abs(input) < C1*eu:
            return input/abs(input)*(fa+Esh*(abs(input)-esh))
        elif abs(input) < eu:
            return input/abs(input)*(fc1eu + (fu-fc1eu)/(eu-C1*eu)*(abs(input)-C1*eu))
        else:
            return fu

    def strainToRebarStress(input):
        if abs(input) < fs/Es:
            return input*Es
        else:
            return abs(input)/input*fs

    df['Profile Area'] = df['Fiber Coordinates'].apply(
        coordinateToSteelArea, args=(axis,))
    df['Rebar Area'] = df['Fiber Coordinates'].apply(
        coordinateToRebarArea, args=(shape,)).fillna(0)
    df['Concrete Area'] = df['Fiber Coordinates'].apply(
        coordinateToGrossArea, args=(shape,)) - df['Profile Area'] - df['Rebar Area']

    noNs = 30
    Nmax = -strainToConcreteStress(-0.0035)*df['Concrete Area'].sum(
    ) + fa*df['Profile Area'].sum() + fs*df['Rebar Area'].sum()
    N = np.zeros([noNs+1, 2])
    for idx, x in enumerate(N):
        x[0] = Nmax/noNs*idx

    for y in N:
        def summer(x):
            def coordinateToStrain(input):
                return -ecu1*(x[0]-input)/x[0]

            df['Strains'] = df['Fiber Coordinates'].apply(coordinateToStrain)
            df['Concrete Stress'] = df['Strains'].apply(strainToConcreteStress)
            df['Concrete Force'] = df['Concrete Stress']*df['Concrete Area']
            df['Rebar Stress'] = np.where(
                df['Rebar Area'] != 0, df['Strains'].apply(strainToRebarStress), 0)
            df['Rebar Force'] = df['Rebar Stress']*df['Rebar Area']
            df['Profile Stress'] = np.where(
                df['Profile Area'] != 0, df['Strains'].apply(strainToProfileStress), 0)
            df['Profile Force'] = df['Profile Stress']*df['Profile Area']

            return df['Concrete Force'].sum() + df['Rebar Force'].sum() + df['Profile Force'].sum() + y[0]

        sol = scipy.optimize.root(summer, 1, method='hybr')
        df['Fiber Coordinates from Centroid'] = df['Fiber Coordinates'] - depth/2
        y[1] = (df['Fiber Coordinates from Centroid'] *
                (df['Concrete Force']+df['Rebar Force']+df['Profile Force'])).sum()
        print(sol.x, sol.message, sol.fun)

# %%

    df_MN = pd.DataFrame(N, columns=['N', 'M'])
    return df_MN


if __name__ == '__main__':
    df_MN = main()
