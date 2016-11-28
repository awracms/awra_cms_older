

def lower_dict(in_dict):
    res = {}
    for k,v in list(in_dict.items()):
        res[k.lower()] = v
    return res

def map_dict(base_dict, new_dict):
    for k,v in list(new_dict.items()):
        if k in base_dict:
            base_dict[k] = v
    return base_dict
