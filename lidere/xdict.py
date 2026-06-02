from functools import partial

class xdict(dict):
    """ 
    An extended dictionary that allows access to elements as atttributes and counts 
    these accesses. This way, we know if some attributes were never used. 
    """

    def __init__(self, *args, **kwargs):
        from collections import Counter
        super().__init__(*args, **kwargs)
        
        self.__dict__['__counter__'] = Counter()

        
        def _a(x, x_orig):
            if isinstance(x_orig, xdict):
                return x_orig.__class__(x)
            else:
                return self.__class__(x)

        for k in self.keys():
            self[k] = self.map_dicts(_a, x=self[k])

    def __getitem__(self, k):
        self.__dict__['__counter__'][k] += 1
        return super().__getitem__(k)

    def __getattr__(self, k):
        self.__dict__['__counter__'][k] += 1
        return super().get(k)

    def __setattr__(self, k, v):
        return super().__setitem__(k, v)

    def __delattr__(self, k):
        return super().__delitem__(k)

    def items(self):
        return ((k, v) for k, v in super().items())

    def unused_keys(self, exceptions=()):
        return [k for k in super().keys() if self.__dict__['__counter__'][k] == 0 and k not in exceptions]

    def assume_no_unused_keys(self, exceptions=()):
        if len(self.unused_keys(exceptions=exceptions)) > 0:
            raise ValueError(f'Unused keys: {self.unused_keys(exceptions=exceptions)}')

    @staticmethod
    def merge(this, other):
        out = xdict(other) 
        for k, value in this.items():
            if k in other:
                if isinstance(value, dict) and isinstance(other[k], dict):
                    out[k] = xdict.merge(value, other[k])
                else:
                    out[k] = value
            else:
                out[k] = value
        return out
    
    def __add__(this, other):
        return xdict.merge(this, other)    

    def init(self, recursive=True):

        import importlib

        def _init(x, depth=0, recursive=True):

            if type(x) in {list, tuple} and recursive:
                return [_init(a, depth=depth+1, recursive=recursive) for a in x]

            if isinstance(x, dict):

                # prevent initialization of children of __no_init__
                if ('__no_init__' in x and x['__no_init__']) and depth>0:
                    return x
                
                x2 = xdict()
                
                # first recurse into depth
                for k in x.keys():
                    if (isinstance(x[k], dict) or type(x[k]) in {list, tuple}) and recursive:
                        x2[k] = _init(x[k], depth=depth+1, recursive=recursive)
                    else:
                        x2[k] = x[k]

                if '__class__' in x:
                    
                    if isinstance(x['__class__'], str):
                        module = importlib.import_module('.'.join(x['__class__'].split('.')[:-1]))
                        attr = getattr(module, x['__class__'].split('.')[-1])
                    else:
                        attr = x['__class__']
                    
                    # the __no_init__ flag blocks unless we are at depth=0, i.e. init was called on this object
                    if ('__no_init__' not in x or not x['__no_init__']) or depth == 0:
                        args = {k: v for k,v in x2.items() if k != '__class__' and k!= '__no_init__'}
                        return attr(**args)

                return x2
        
            return x
        
        obj = _init(xdict(self), recursive=recursive)

        return obj

    def to_dict(self):
        return self.map_dicts(lambda x, _: dict(x), None, is_first=True)

    def map(self, f):
        """ apply function f to each element. """
        
        def _map(x):

            if type(x) in {list, tuple}:
                return [_map(a) for a in x]

            if isinstance(x, dict):
                x2 = xdict()
                
                for k, v in x.items():
                    if isinstance(v, dict):
                        x2[k] = _map(v)
                    else:
                        x2[k] = f(k, v)

                return x2

            return x
        
        obj = _map(self)
        return obj
    
    def map_dicts(self, f, x, is_first=False):
        if is_first:
            x = self

        if type(x) in {list, tuple}:
            return [self.map_dicts(f, a) for a in x]  
          
        if isinstance(x, dict):
            return f({k: self.map_dicts(f, v) for k, v in x.items()}, x)

        return x

    def serialize(self):

        def f(k, v):
            if isinstance(v, type):
                return v.__name__
            if isinstance(v, partial):
                return dict(__class__=v.func, **v.keywords)
            else:
                return v

        return self.map(f)


    def dump_yaml(self, filename):
        import yaml
        import os

        if os.path.dirname(filename) != '':
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        with open(filename, 'w') as fh:
            yaml.safe_dump(dict(self.to_dict()), fh)

    def hash(self):
        import hashlib
        import base64

        #return int.from_bytes(hashlib.sha384(str(self).encode('utf8')).digest(), 'big')
        h =  hashlib.sha1(str(self).encode('utf8')).digest()
        return base64.urlsafe_b64encode(h).decode('utf8')[:12]
        
    @staticmethod
    def flatten(d, prefixes=()):
        out = dict()
        for k, v in d.items():
            if isinstance(v, dict):
                out.update(xdict.flatten(v, prefixes=prefixes+(k,)))
            else:
                k2 = '.'.join(prefixes+('',)) +  k
                out[k2] = v

        return out


class xdictS(xdict):

    def __init__(self, *args, **kwargs):
        from collections import Counter
        super().__init__(*args, **kwargs)
    
        self['__no_init__'] = True
        



if __name__ == '__main__':

    # tests

    class A:
        def __init__(self, x, y=2):
            self.is_init = True

    # init from dict
    a = xdict(dict(__class__=A, x=3))
    assert type(a) == xdict
    a.init()
    assert not a.is_init
        
    a = xdict(b=xdictS(__class__=A, x=2, y=3)).init()
    assert not a.b.is_init

    # by default, initialize objects at all hierarchies
    a = xdict(a=xdict(__class__=A, x=2)).init()
    assert a.a.is_init

    # ...unless recursive is set to False
    a = xdict(a=xdict(__class__=A, x=2)).init(recursive=False)
    assert not a.a.is_init    

    # ... or the object is xdictS
    a = xdict(a=xdictS(__class__=A, x=2)).init()
    assert not a.a.is_init    

    a = xdictS(__class__=A, x=2).init()
    assert a.is_init
    
    # running init explicitly on a xdictS will initialize the object, though.
    a = xdictS(a=A(x=2)).init()
    assert a.a.is_init

    # every sub dict will become an xdict
    a = xdict(a=dict(__class__=A, x=2))
    assert type(a.a) == xdict

    # no change
    a = xdictS(a=A(x=2))
    a.map(lambda k, v: v)

    a.to_dict()

    