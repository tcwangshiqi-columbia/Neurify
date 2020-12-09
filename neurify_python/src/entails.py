import ctypes
import numpy as np

libname = "/home/rhiba/packages/lib/libentails.so"
c_lib = ctypes.CDLL(libname)

intp = ctypes.POINTER(ctypes.c_int)
floatp = ctypes.POINTER(ctypes.c_float)

c_lib.entails.restype = ctypes.c_int
c_lib.entails.argtypes = [intp,ctypes.c_int,floatp,ctypes.c_int,floatp,floatp,ctypes.c_char_p]

#int entails( int *h, int h_size, float *input, int input_size, float *u_bounds, float *l_bounds, char* network_path);
def entails(h,h_size,inp,inp_size,u_bounds,l_bounds,network_path):
    # needed for c_char_p type
    network_path = network_path.encode('utf-8')

    h = np.array(h,dtype=ctypes.c_int)
    inp = np.array(inp,dtype=ctypes.c_float)
    u_bounds = np.array(u_bounds,dtype=ctypes.c_float)
    l_bounds = np.array(l_bounds,dtype=ctypes.c_float)

    h_c = h.ctypes.data_as(intp)
    inp_c = inp.ctypes.data_as(floatp)
    u_bounds = u_bounds.ctypes.data_as(floatp)
    l_bounds = l_bounds.ctypes.data_as(floatp)
    
    res = c_lib.entails(h_c,h_size,inp_c,inp_size,u_bounds,l_bounds,network_path)
    return res


'''
if __name__ == "__main__":
    h = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,25,26,27,28,29,45,46,47,48,49]
    h_size = len(h)
    inp = [0.5,0.5,0.5,0.5,0.5,0.6158970519900322,0.6500097215175629,0.5419477485120296,0.5759362429380417,0.6085240393877029,0.6028607115149498,0.5812925100326538,0.5288058891892433,0.5238163638859987,0.5813853070139885,0.5316681489348412,0.5259015411138535,0.5261953305453062,0.5069217910058796,0.5252024382352829,0.6301251798868179,0.6642986238002777,0.5501794219017029,0.6265065670013428,0.6569506227970123,0.4825815800577402,0.43560905009508133,0.4821596648544073,0.42361055314540863,0.45672742277383804,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
    inp_size = len(inp)
    eps = 0.05
    net_path = "models/SST_fc_5d_10inp_format_16hu_norm.nnet"
    is_adv = entails(h,h_size,inp,inp_size,eps,net_path)
    print("is adv:",is_adv)

    h = [0,1,2,3,4]
    h_size = len(h)
    is_adv = entails(h,h_size,inp,inp_size,eps,net_path)
    print("is adv:",is_adv)
'''
