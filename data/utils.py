

def calibrate_etco2(value, machine_name):
    """_summary_

    Args:
        value (_type_): _description_
        machine_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    return value if machine_name != "Datex-Ohmeda/ETCO2" else 7.5 * value