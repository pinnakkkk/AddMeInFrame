


import pydoc


def object_from_dict(dict, parent=None, **default_kwargs):
    kwargs = dict.copy()
    obj_type = kwargs.pop("type")
    
    for key, value in default_kwargs.items():
        kwargs.setdefault(key, value)    
    
    if parent is not None:
        return getattr(parent, obj_type)(**kwargs)
        
    obj = pydoc.locate(obj_type)
    
    if obj is None:
        raise ImportError(f"Could not locate {obj_type}")

    return obj(**kwargs)