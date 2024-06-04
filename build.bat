@echo off
gcc -ggdb -O3 -mavx2 -mfma -Wall -Wextra -Wpedantic -Wno-missing-braces Renderer.c main.c -o main.exe -lgdi32 -lwinmm
