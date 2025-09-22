

def polyak_avg(dict_1, dict_2, tau=0.999):
    for key in dict_1.keys():
        dict_1[key] = tau*dict_1[key] + (1-tau)*dict_2[key]
    return dict_1