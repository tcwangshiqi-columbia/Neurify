@echo off

set c=cl

REM determine platform (win32/win64)
echo main(){printf("SET PLATFORM=win%%d\n", (int) (sizeof(void *)*8));}>platform.c
%c% /nologo platform.c /Feplatform.exe
del platform.c
platform.exe >platform.bat
del platform.exe
call platform.bat
del platform.bat

rem Implicit Dynamic linked with the lpsolve library
%c% -DWIN32 /Zp8 /Gd -I..\.. demo.c implicit.c ..\..\lpsolve55\bin\%PLATFORM%\lpsolve55.lib -o demoi

rem Explicit Dynamic linked with the lpsolve library
%c% -DWIN32 -DEXPLICIT /Zp8 /Gd -I..\.. demo.c explicit.c -o demoe

rem Static linked with the lpsolve library
%c% -DWIN32 /Zp8 /Gd /MT -I..\.. demo.c implicit.c ..\..\lpsolve55\bin\%PLATFORM%\liblpsolve55.lib -o demos

set PLATFORM=
