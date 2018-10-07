@echo off

set c=bcc32

REM determine platform (win32/win64)
echo main(){printf("SET PLATFORM=win%%d\n", (int) (sizeof(void *)*8));}>platform.c
%c% platform.c -eplatform.exe
del platform.c
platform.exe >platform.bat
del platform.exe
call platform.bat
del platform.bat

rem Implicit Dynamic linked with the lpsolve library
rem %c% -w-8057 -DWIN32 /O2 -a8 -I..\.. -edemoi.exe demo.c implicit.c ..\..\lpsolve55\bin\%PLATFORM%\lpsolve55.lib

rem Explicit Dynamic linked with the lpsolve library
%c% -w-8057 -DWIN32 -DEXPLICIT /O2 -a8 -I..\.. -edemoe.exe demo.c explicit.c

rem Static linked with the lpsolve library
rem %c% -w-8057 -DWIN32 /O2 -a8 -I..\.. -edemos.exe demo.c implicit.c ..\..\lpsolve55\bin\%PLATFORM%\liblpsolve55.lib

set PLATFORM=
