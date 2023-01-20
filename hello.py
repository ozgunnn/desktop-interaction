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
    NM = pyfuncs.main(float(fcm), float(fa), float(fs), float(gam_fcm), float(gam_fa), float(gam_fs), float(h),
                      float(b), float(tw), float(tf), float(depth), int(reb_no), float(reb_d), axis, shape)
    N = NM['N']
    M = NM['M']
    print(N, M)
    print(fcm, fa, fs, gam_fcm, gam_fa, gam_fs, h, b,
          tw, tf, depth, reb_no, reb_d, axis, shape)
    return N, M


say_hello_py('Python World!')
eel.say_hello_js('Python World!')   # Call a Javascript function


options = {
    'mode': 'custom',
    'args': ['node_modules/electron/dist/electron.exe', '.']
}

eel.start('hello.html')
#eel.start('hello.html', mode='custom', cmdline_args=['node_modules/electron/dist/electron.exe', '.'])
