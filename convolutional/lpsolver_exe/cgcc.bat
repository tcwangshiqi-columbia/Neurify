@echo off

set c=gcc

REM determine platform (win32/win64)
echo main(){printf("SET PLATFORM=win%%d\n", (int) (sizeof(void *)*8));}>platform.c
%c% platform.c -o platform.exe
del platform.c
platform.exe >platform.bat
del platform.exe
call platform.bat
del platform.bat

%c% -DWIN32 -DEXPLICIT -O3 -I..\.. demo.c explicit.c -o demoe
%c% -DWIN32 -O3 -I..\.. demo.c implicit.c ..\..\lpsolve55\bin\%PLATFORM%\lpsolve55.lib -o demoi

set PLATFORM=
