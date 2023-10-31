import math
import random

SamplingInterval = None # Sampling interval - defaults to None, must be set explicitly

# ============================================================
# Controllers

'''
    Include your controller(s) code here
'''



# ============================================================
# # Input signals

'''
    If you test your target system with various input signals, include
    the code here. You will only invoke the input signal that you 
    finalized, when you submit your code. 
'''

# # ============================================================
# # Loop functions

def static_test( target_ctor, ctor_args, umax, steps, repeats, tmax ):
    # Complete test for static process characteristic
    print("Inside FB ST")
    for i in range( 0, steps ):
        u = float(i)*umax/float(steps)

        for r in range( repeats ):
            target = target_ctor(*ctor_args)

            for t in range( tmax ):
                y = target.work(u)

            print(u, y)
                        
    quit()


def closed_loop():
    '''
        Your feedback loop code goes here
    '''
    quit()