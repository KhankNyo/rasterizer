@echo off
gcc -mtune=skylake -flto -Ofast -mbmi -mavx2 -mfma -Wall -Wextra -Wpedantic -Wno-missing-braces Renderer.c main.c -o main.exe -lgdi32 -lwinmm


