"""Windows user-bound DPAPI credentials; never add a password to argv/config/API."""
import ctypes,os
from ctypes import wintypes
from .storage import ProjectError

class Blob(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('data',ctypes.POINTER(ctypes.c_ubyte))]

def crypt(raw,encrypt=True):
    if os.name!='nt':raise ProjectError(503,'SSH_CREDENTIAL_STORAGE_UNAVAILABLE','密码保存需要Windows用户加密服务；可继续使用原免密连接。')
    buffer=ctypes.create_string_buffer(raw);source=Blob(len(raw),ctypes.cast(buffer,ctypes.POINTER(ctypes.c_ubyte)));result=Blob()
    library=ctypes.WinDLL('crypt32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    function=library.CryptProtectData if encrypt else library.CryptUnprotectData
    function.argtypes=[ctypes.POINTER(Blob),ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(Blob)]
    function.restype=wintypes.BOOL;kernel.LocalFree.argtypes=[ctypes.c_void_p];kernel.LocalFree.restype=ctypes.c_void_p
    if not function(ctypes.byref(source),None,None,None,None,1,ctypes.byref(result)):
        raise ProjectError(503,'SSH_CREDENTIAL_UNAVAILABLE','无法加密或解密此Windows用户的SSH密码，请重新输入。')
    try:return ctypes.string_at(result.data,result.size)
    finally:kernel.LocalFree(result.data)

def protect(password):return crypt(password.encode('utf-8'))
def unprotect(path):
    try:return crypt(path.read_bytes(),False).decode('utf-8')
    except OSError as exc:raise ProjectError(503,'SSH_CREDENTIAL_UNAVAILABLE','已保存的SSH密码不可用，请重新配置连接。') from exc
