#ifndef COMMON_H
#define COMMON_H

#include <stdint.h>

typedef uint8_t u8;
typedef uint16_t u16;
typedef int16_t i16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef size_t uSize;
typedef intptr_t iSize;
typedef uint8_t Bool8;
#define false 0
#define true 1

#define IN_RANGE(lower, n, upper) ((lower) <= (n) && (n) <= (upper))
#define SWAP(typename, a, b) do {\
    typename t = a;\
    a = b;\
    b = t;\
} while (0)


typedef struct platform_state platform_state;
Bool8 Platform_IsKeyPressed(const platform_state *State, char Key);
Bool8 Platform_IsKeyDown(platform_state *State, char Key, double Delay);
double Platform_GetTimeMillisec(void);
void Platform_DeallocateMemory(void *Ptr);
void *Platform_AllocateMemory(uSize SizeBytes);
#define Platform_AllocateArray(member_type, member_count) \
    Platform_AllocateMemory((member_count) * sizeof(member_type))


typedef union v3i
{
    int Index[3];
    struct {
        int x, y, z;
    };
} Face, v3i;
typedef union v3f
{
    float Index[4];
    struct {
        float x, y, z, Dummy;
    };
} v3f;

typedef struct 
{
    u32 FaceCount, FaceCapacity;
    Face *Faces;
    u32 *Colors;
    Bool8 ColorIsValid;

    uSize VertexCount, VertexCapacity;
    v3f *Vertices;
} obj_model;

typedef struct 
{
    u32 *Buffer;
    float *ZBuffer;
    u32 Width, Height;
    uSize ZBufferCapacity;
} renderer_context;

typedef struct
{
    char *FileBuffer;
    uSize FileSize;
    obj_model Model;
    u32 RandState;

    renderer_context RenderContext;

    v3f Light;
    double LightMoveStart;
    double LightMoveDelta;
    double LightDeltaX;
} app_state;

app_state App_OnStartup(void);
void App_OnLoop(app_state *AppState, platform_state *PlatformState);
void App_OnPaint(app_state *AppState, u32 *Buffer, u32 Width, u32 Height);
/* app now owns the ptr */
void App_OnFileReceived(app_state *AppState, void *FileBuffer, uSize FileSize);

#endif /* COMMON_H */

