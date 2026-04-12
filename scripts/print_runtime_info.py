import os
import sys

def main() -> None:
    print('exe:', sys.executable)
    print('prefix:', sys.prefix)
    print('base_prefix:', getattr(sys, 'base_prefix', None))
    print('path:')
    for p in sys.path:
        print('  ', p)

    import urllib
    import urllib.request

    print('urllib:', urllib.__file__)
    print('urllib.request:', urllib.request.__file__)

    for k in ['PYTHONHOME', 'PYTHONPATH', 'CONDA_PREFIX', 'VIRTUAL_ENV']:
        print(f'{k}:', os.environ.get(k))

if __name__ == '__main__':
    main()
