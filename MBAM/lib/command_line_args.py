import argparse

def GetParser():

    '''
    Function to define command line arguments which are used across multiple scripts
    '''

    parser = argparse.ArgumentParser(
    description='Reduce an ion channel model using the MBAM')
    parser.add_argument("-d", "--done", action='store_true', help="whether geodesic has been done or not",
        default=False)
    parser.add_argument("-i", "--invert", action='store_true', help="whether to invert initial velocity or not",
        default=False)
    parser.add_argument('--ssv_threshold', type=float, default=1e-3, 
        help='what threshold to use for the smallest singular value')
    parser.add_argument("-p", "--plot", action='store_true', help="whether to show plots or not",
        default=False)
    parser.add_argument("-g", "--gamma", action='store_true', help="whether to use 2nd order sensitivities or not \
        (approximates this using finite differences by default)",
        default=False)
    parser.add_argument("--show_rates", action='store_true', help="whether to show transition rates or not",
        default=False)
    parser.add_argument('--eig', type=int, default=1, 
        help='which initial eigendirection to use')
    
    return parser