import eel
import pandas
import pyfuncs
# Set web files folder
eel.init('web')


@eel.expose                         # Expose this function to Javascript
def say_hello_py(x):
    print('Hello from %s' % x)


@eel.expose
def calculate_arrays(a, b):
    print(a*b)


@eel.expose
def calculate(fcm, fa, fs, gam_fcm, gam_fa, gam_fs, h, b, tw, tf, depth, reb_no, reb_d, axis, shape):
    NM, NM1, NM2 = pyfuncs.main(float(fcm), float(fa), float(fs), float(gam_fcm), float(gam_fa), float(gam_fs), float(h),
                                float(b), float(tw), float(tf), float(depth), int(reb_no), float(reb_d), axis, shape)
    N = NM['N']/1000
    M = NM['M']/1000000
    N1 = NM1['N']/1000
    M1 = NM1['M']/1000000
    N2 = NM2['N']/1000
    M2 = NM2['M']/1000000
    # print(type(N))
    return N.tolist(), M.tolist(), N1.tolist(), M1.tolist(), N2.tolist(), M2.tolist()


@eel.expose
def printpy(*arg):
    print(arg)


say_hello_py('Python World!')
eel.say_hello_js('Python World!')   # Call a Javascript function


options = {
    'mode': 'custom',
    'args': ['node_modules/electron/dist/electron.exe', '.']
}

eel.start('hello.html')
#eel.start('hello.html', mode='custom', cmdline_args=['node_modules/electron/dist/electron.exe', '.'])
