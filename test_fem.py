import ctypes as ct

libc = ct.cdll.LoadLibrary("./libmyadd.so")

if (__name__ == "__main__"):
    print(libc.main())
