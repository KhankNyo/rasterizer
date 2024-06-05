
#include <stdio.h>
#include <string.h> /* memcpy */
#include <math.h>   /* pow */
#include <float.h>  /* FLT_MIN */

#include <immintrin.h>
#include "Common.h"

#define ABS(v) (((v) < 0)? -(v) : (v))
#define MIN(a, b) ((a) < (b)? (a): (b))
#define MAX(a, b) ((a) > (b)? (a): (b))
#define APP_RAND(app_state_ptr) \
    ((((app_state_ptr)->RandState = (214013*(app_state_ptr)->RandState + 2531011)) >> 16) & 0x7FFF)
#define TRIANGLE_DBG 0
#define SSE_ALIGN __attribute__((aligned(16)))
#define AVX_ALIGN __attribute__((aligned(32)))
#define FORCE_INLINE __attribute__((always_inline)) inline

typedef union v2i 
{
    int Index[2];
    struct {
        int x, y;
    };
    struct {
        int u, v;
    };
} v2i;


static v3f V3f_Sub(v3f A, v3f B);
static v3f V3f_NormalizedCrossProd(v3f A, v3f B);
static v3f V3f_Normalize(v3f Vec);
float V3f_DotProd(v3f A, v3f B);


static u32 RGBColor(u8 R, u8 G, u8 B)
{
    return (u32)R << 16 | (u32)G << 8 | B;
}

static FORCE_INLINE int Roundf(float f)
{
    return (int)(f + .5);
}

static FORCE_INLINE float Recip(float f)
{
    __m128 v = _mm_load_ss(&f);
    _mm_store_ss(&f, _mm_rcp_ss(v));
    return f;
}


static void App_DrawLine(renderer_context *Context, int x0, int y0, int x1, int y1, u32 Color)
{
    Bool8 Steep = false;
    if (ABS(x0 - x1) < ABS(y0 - y1)) /* transpose the image if it's steep */
    {
        SWAP(int, x0, y0);
        SWAP(int, x1, y1);
        Steep = true;
    }
    if (x0 > x1)
    {
        SWAP(int, x0, x1);
        SWAP(int, y0, y1);
    }
    int Dx = x1 - x0;
    int Dy = y1 - y0;
    int Derr2 = ABS(Dy)*2;
    int Err2 = 0;
    int y = y0;

    for (int x = x0; x <= x1; x++)
    {
        if (Steep)
        {
            if (IN_RANGE(0, y, (int)Context->Width - 1) 
            && IN_RANGE(0, x, (int)Context->Height - 1))
            {
                int Index = x*Context->Width + y;
                Context->Buffer[Index] = Color;
            }
        }
        else
        {
            if (IN_RANGE(0, x, (int)Context->Width - 1) 
            && IN_RANGE(0, y, (int)Context->Height - 1))
            {
                int Index = y*Context->Width + x;
                Context->Buffer[Index] = Color;
            }
 
        }

        Err2 += Derr2;
        if (Err2 > Dx)
        {
            y += (y1 > y0)? 1 : -1;
            Err2 -= Dx*2;
        }
    }
}

static void App_DrawHorizontalLine(renderer_context *Context, int x0, int y, int x1, u32 Color)
{
    if (y >= (int)Context->Height) 
    {
        return;
    }
    if (x1 < x0)
    {
        SWAP(int, x1, x0);
    }
    x1 = MIN(x1, (int)Context->Width);

    int RemSize = x1 - x0 + 1;
    u32 *Ptr = Context->Buffer + y*Context->Width + x0;

    while (RemSize > 0 && (uintptr_t)Ptr % 32)
    {
        *Ptr++ = Color;
        RemSize--;
    }

    __m256i ColorVec = _mm256_set1_epi32(Color);
    while (RemSize > 7)
    {
        _mm256_store_si256((__m256i *)Ptr, ColorVec);
        Ptr += 8;
        RemSize -= 8;
    }

    while (RemSize > 0)
    {
        *Ptr++ = Color;
        RemSize--;
    }
}



/* A is the pointy end, B and C are base points */
static void App_Draw3DHorizontalSideTriangle(
    renderer_context *Context, 
    v3i A, v3i B, v3i C,
    u32 Color)
{
    int YStart = A.y;
    int YEnd = C.y;
    int Height = A.y - C.y;
    if (Height == 0)
        Height = 1;

    float Heightf = Height;
    float XBegin = A.x;
    float XEnd = A.x;

    float DeltaLeft;
    float DeltaRight;
    float DeltaZ_AB; 
    float DeltaZ_AC;
    {
        float Out[4];
        __m128 Tmp = _mm_set_ps(
            A.x - B.x, 
            C.x - A.x, 
            B.z - A.z, 
            C.z - A.z
        );
        __m128 HeightfVec = _mm_set1_ps(Heightf);
        __m128 Result = _mm_mul_ps(Tmp, _mm_rcp_ps(HeightfVec)); /* divide individual values by Heightf */
        _mm_store_ps(Out, Result);

        DeltaLeft = Out[3];
        DeltaRight = Out[2];
        DeltaZ_AB = Out[1];
        DeltaZ_AC = Out[0];
    }

    float Z_AB = A.z;
    float Z_AC = A.z;
    if (YStart < YEnd)
    {
        SWAP(int, YStart, YEnd);
        XBegin = B.x;
        XEnd = C.x;

        Z_AB = B.z;
        Z_AC = C.z;
    }

    float Width = Context->Width;
    for (int y = YStart;
        y >= YEnd; 
        y--, 
        XEnd += DeltaRight, 
        XBegin -= DeltaLeft)
    {
        if (XBegin >= Width || XEnd < 0 || y >= (int)Context->Height || y < 0)
            continue;

        if (XEnd > Width)
            XEnd = Width;
        if (XBegin < 0)
            XBegin = 0;

        int YIndex = y * Context->Width;
        int Start = YIndex + Roundf(XBegin);
        int End = YIndex + Roundf(XEnd);
        if (Start > End)
        {
            SWAP(int, Start, End);
        }
        float Len = XEnd - XBegin + 1;
        float RecipLen = Recip(Len);
        float Dz = Z_AB*RecipLen - Z_AC*RecipLen;
        float z = Z_AC;
        Z_AB += DeltaZ_AB;
        Z_AC += DeltaZ_AC;

        for (int i = Start; i <= End; i += 1)
        {
            if (z > Context->ZBuffer[i])
            {
                Context->ZBuffer[i] = z;
                Context->Buffer[i] = Color;
            }
            z += Dz;
        }
    }
}

static void App_Draw3DTriangle(renderer_context *Context, v3i A, v3i B, v3i C, u32 Color)
{
    if (B.y > A.y)
        SWAP(v3i, A, B);
    if (C.y > A.y)
        SWAP(v3i, A, C);
    /* A is now the topmost point */
    if (B.y < C.y)
        SWAP(v3i, B, C);
    /* C is now the bottom most point */

    int Dy = A.y - C.y;
    int Dx = A.x - C.x;
    int Dz = A.z - C.z;

    float Mx = C.x;
    if (Dy != 0)
    {
        float DxDy = Dx * Recip(Dy);
        Mx += B.y*DxDy - C.y*DxDy; /* allow the compiler to use fma */
        //Mx += (B.y - C.y) * Recip(Dy) * Dx;
    }

    float Mz = C.z;
    if (Dz != 0)
    {
        float RecipDz = Recip(Dz);
        Mz += B.z*RecipDz - C.z*RecipDz; /* allow the compiler to use fma */
        //Mz += (B.z - C.z)*RecipDz;
    }
    
    v3i M = {
        .x = Roundf(Mx), 
        .y = B.y,
        .z = Roundf(Mz),
    };

    App_Draw3DHorizontalSideTriangle(
        Context, 
        A, B, M,
        Color
    );
    B.y -= 1;
    M.y -= 1;
    App_Draw3DHorizontalSideTriangle(
        Context,
        C, B, M,
        Color
    );
    

#if TRIANGLE_DBG
    u32 DbgColor = RGBColor(0x80, 0x80, 0);
    App_DrawLine(Context, A.x, A.y, B.x, B.y, DbgColor);
    App_DrawLine(Context, C.x, C.y, B.x, B.y, DbgColor);
    App_DrawLine(Context, A.x, A.y, C.x, C.y, DbgColor);
#endif
}

static void App_Draw2DTriangle(renderer_context *Context, v2i A, v2i B, v2i C, u32 Color)
{
    if (B.y > A.y)
        SWAP(v2i, A, B);
    if (C.y > A.y)
        SWAP(v2i, A, C);
    /* A is now the topmost point */
    if (B.y < C.y)
        SWAP(v2i, B, C);
    /* C is now the bottom most point */

    int Dy = A.y - C.y;
    int Dx = A.x - C.x;
    if (Dy == 0)
    {
        /* since A is the topmost and C is the bottom most point, 
         * having a Dy == 0 means all 3 points lie on the same line, 
         * draw them as a line */
        if (B.x < A.x)
            SWAP(v2i, A, B);
        if (C.x < A.x)
            SWAP(v2i, A, C);
        /* A is now the leftmost point */
        if (B.x > C.x)
            SWAP(v2i, B, C);
        /* C is now the rightmost point */
            
        App_DrawHorizontalLine(Context, A.x, A.y, B.x, Color);
        return;
    }

    /* split the triangle at B by creating a point M on AC (M is parallel to the x axis) */
    //    A 
    //    |\
    //    | \
    //    |  \
    //    |   \
    //    |    \
    //    M-----B
    //    |    /
    //    |   /
    //    |  /
    //    | /
    //    |/
    //    C
    v2i M = {
        .y = B.y,
        .x = Roundf(C.x + (B.y - C.y) / (float)Dy * Dx),
    };


    int Dh = A.y - M.y;
    int DwLeft = A.x - B.x;
    int Width = M.x - B.x;

    float DeltaLeft = (float)DwLeft / (float)Dh;
    float LeftRightSlope = (float)Width / (float)Dh;
    float x = A.x;

    /* top */
    float Len = 0;
    for (int y = A.y; y > M.y; y--)
    {
        App_DrawHorizontalLine(Context, Roundf(x), y, Roundf(x + Len), Color);
        Len += LeftRightSlope;
        x -= DeltaLeft;
    }

    /* bottom */
    Dh = M.y - C.y;
    DwLeft = B.x - C.x;

    DeltaLeft = (float)DwLeft / (float)Dh;
    LeftRightSlope = -(float)Width / (float)Dh;
    x = B.x;
    Len = Width;
    for (int y = M.y; y >= C.y; y--)
    {
        App_DrawHorizontalLine(Context, Roundf(x), y, Roundf(x + Len), Color);
        Len += LeftRightSlope;
        x -= DeltaLeft;
    }

#if TRIANGLE_DBG
    u32 DbgColor = RGBColor(0x80, 0x80, 0);
    App_DrawLine(Context, A.x, A.y, B.x, B.y, DbgColor);
    App_DrawLine(Context, C.x, C.y, B.x, B.y, DbgColor);
    App_DrawLine(Context, A.x, A.y, C.x, C.y, DbgColor);
#endif
}

static void App_SetBgColor(renderer_context *Context, u32 Color)
{
    u32 BufferSize = Context->Width*Context->Height;
    {
        u32 *Ptr = Context->Buffer;
        u32 i = 0; 
        while ((uintptr_t)Ptr % 32 && i < BufferSize)
        {
            *Ptr++ = Color;
            i++;
        }
        BufferSize -= i;

        __m256i ColorVec = _mm256_set1_epi32(Color);
        /* writing out of bound is ok on x86 if it's in page boundary */
        for (u32 i = 0; i < BufferSize; i += 8)
        {
            _mm256_store_si256((__m256i *)Ptr, ColorVec);
            Ptr += 8;
        }
    }
    
    {
        uSize BufferSize = Context->Width * Context->Height;
        if (BufferSize > Context->ZBufferCapacity)
        {
            Context->ZBufferCapacity = BufferSize*2 + 8;
            if (Context->ZBuffer)
            {
                Platform_DeallocateMemory(Context->ZBuffer);
            }
            Context->ZBuffer = Platform_AllocateArray(*Context->ZBuffer, Context->ZBufferCapacity);
        }


        float *Ptr = Context->ZBuffer;
        uSize i = 0;
        while ((uintptr_t)Ptr % 32 && i < BufferSize)
        {
            *Ptr++ = -FLT_MAX;
            i++;
        }
        BufferSize -= i;

        __m256 MinVec = _mm256_set1_ps(-FLT_MAX);
        for (uSize i = 0; i < BufferSize; i += 8)
        {
            _mm256_store_ps(Ptr, MinVec);
            Ptr += 8;
        }
    }
}


static int App_ParseInt(const char *Str, int Len, int *OutIndex)
{
    int Ret = 0;
    if (Len == 0)
    {
        *OutIndex = 0;
        return 0;
    }


    Bool8 Neg = Str[0] == '-';
    int i = Neg;
    while (i < Len && IN_RANGE('0', Str[i], '9'))
    {
        Ret *= 10;
        Ret += Str[i] - '0';
        i++;
    }
    *OutIndex = i;
    return Neg? -Ret : Ret;
}

static float App_ParseFloat(const char *Str, int Len, int *OutIndex)
{
    float Ret = 0;
    if (Len == 0)
    {
        *OutIndex = 0;
        return 0;
    }

    Bool8 Neg = Str[0] == '-';
    int i = Neg;
    while (i < Len && IN_RANGE('0', Str[i], '9'))
    {
        Ret *= 10;
        Ret += Str[i] - '0';
        i++;
    }

    if (i < Len && '.' == Str[i])
    {
        i++; /* skip '.' */
        float Fraction = 0;
        float Power = 1;
        while (i < Len && IN_RANGE('0', Str[i], '9'))
        {
            Fraction *= 10;
            Power *= 10;
            Fraction += Str[i] - '0';
            i++;
        }

        Ret += Fraction / Power;
    }

    if (i < Len && 
    ('e' == Str[i] || 'E' == Str[i]))
    {
        i++; /* skip 'e'/'E' */
        int ExpLen;
        int Exponent = App_ParseInt(&Str[i], Len, &ExpLen);
        Ret *= pow(10, Exponent);
        i += ExpLen;
    }

    *OutIndex = i;
    return Neg? -Ret : Ret;
}


static iSize App_SkipSpace(const char *Str, iSize Len)
{
    iSize i = 0;
    while (i < Len && 
    (Str[i] == ' ' || Str[i] == '\t' || Str[i] == '\n' || Str[i] == '\r'))
    {
        i++;
    }
    return i;
}

static void App_ParseObjModel(obj_model *OutModel, const char *File, iSize FileSize)
{
    OutModel->VertexCount = 0;
    OutModel->FaceCount = 0;

    iSize i = 0; 
    while (i < FileSize)
    {
        switch (File[i])
        {
        case '#':
        {
            while (i < FileSize && File[i] != '\n')
                i++;
        } break;
        case 'v':
        {
            /* only care about 'v' */
            if (i + 1 < FileSize && File[i + 1] != ' ')
            {
                i += 2;
                break;
            }

            v3f Vec = { 0 };
            while (i < FileSize && File[i] != '-' && !IN_RANGE('0', File[i], '9'))
            {
                i++;
            }
            for (int j = 0; j < 3 && i < FileSize; j++)
            {
                i += App_SkipSpace(File + i, FileSize - i);

                int Len;
                Vec.Index[j] = App_ParseFloat(File + i, FileSize - i, &Len);
                if (Vec.Index[j] > 1)
                {
                    i += Len - 1 + 1.0;
                }
                i += Len;
            }
            if (OutModel->VertexCount >= OutModel->VertexCapacity)
            {
                OutModel->VertexCapacity = OutModel->VertexCount*2 + 1024;
                void *Ptr = Platform_AllocateArray(*OutModel->Vertices, OutModel->VertexCapacity);

                if (OutModel->Vertices)
                {
                    memcpy(Ptr, OutModel->Vertices, OutModel->VertexCount * sizeof(OutModel->Vertices[0]));
                    Platform_DeallocateMemory(OutModel->Vertices);
                }
                OutModel->Vertices = Ptr;
            }

            OutModel->Vertices[OutModel->VertexCount++] = Vec;
        } break;
        case 'f':
        {
            if (i + 1 < FileSize && File[i + 1] != ' ')
            {
                i += 2;
                break;
            }
            Face Vec = { 0 };
            while (i < FileSize && File[i] != '-' && !IN_RANGE('0', File[i], '9'))
            {
                i++;
            }
            for (int j = 0; j < 3 && i < FileSize; j++)
            {
                i += App_SkipSpace(File + i, FileSize - i);

                int Len;
                Vec.Index[j] = App_ParseInt(File + i, FileSize - i, &Len);
                i += Len;

                /* skip unneeded entries */
                while (i < FileSize && (File[i] != ' ' && File[i] != '\n'))
                    i++;
            }

            /* because the face capacity pointer contains the indeces of the face and color data, 
             * divide capacity by 2 */
            if (OutModel->FaceCount >= OutModel->FaceCapacity/2)
            {
                OutModel->FaceCapacity = OutModel->FaceCount*2 + 1024;
                uSize CapacityInBytes = 
                    OutModel->FaceCapacity/2 * sizeof(OutModel->Faces[0]) 
                    + OutModel->FaceCapacity/2 * sizeof(OutModel->Colors[0]);

                void *Ptr = Platform_AllocateMemory(CapacityInBytes);

                if (OutModel->Faces)
                {
                    memcpy(Ptr, OutModel->Faces, OutModel->FaceCount * sizeof(OutModel->Faces[0]));
                    Platform_DeallocateMemory(OutModel->Faces);
                }
                OutModel->Faces = Ptr;
            }

            OutModel->Faces[OutModel->FaceCount++] = Vec;
        } break;
        default: 
        {
            i++;
        } break;
        }
    }

    OutModel->Colors = (u32 *)(OutModel->Faces + OutModel->FaceCount);
    OutModel->ColorIsValid = false;
}

/* compute A - B */
static v3f V3f_Sub(v3f A, v3f B)
{
    v3f Result;
    _mm_store_ps(Result.Index, 
        _mm_sub_ps(
            _mm_load_ps(A.Index),
            _mm_load_ps(B.Index)
        )
    );
    return Result;
}

/* compute A x B */
static v3f V3f_NormalizedCrossProd(v3f A, v3f B)
{
    __m128 VecA = _mm_load_ps(A.Index);
    __m128 VecB = _mm_load_ps(B.Index);

    __m128 A_yzx = _mm_shuffle_ps(VecA, VecA, 0x09);    /* [Ay, Az, Ax, XX] */
    __m128 B_zxy = _mm_shuffle_ps(VecB, VecB, 0x12);    /* [Bz, Bx, By, XX] */
    __m128 A_zxy = _mm_shuffle_ps(VecA, VecA, 0x12);    /* [Az, Ax, Ay, XX] */
    __m128 B_yzx = _mm_shuffle_ps(VecB, VecB, 0x09);    /* [By, Bz, Bx, XX] */

    __m128 CrossProduct = _mm_fmsub_ps(                 /* [CrossProd vec] */
        A_yzx, B_zxy, 
        _mm_mul_ps(A_zxy, B_yzx)
    );

    __m128 v2 = _mm_mul_ps(CrossProduct, CrossProduct); /* CrossProd vec^2 */
    __m128 x = _mm_shuffle_ps(CrossProduct, CrossProduct, 0x00);    /* distribute x */
    __m128 y = _mm_shuffle_ps(CrossProduct, CrossProduct, 0x55);    /* distribute y */
    __m128 z2 = _mm_shuffle_ps(v2, v2, 0xAA);                       /* get z^2 from crossprod */
    __m128 Tmp = _mm_fmadd_ps(x, x, z2);                            /* sum x^2 and z^2 */
    __m128 Sum = _mm_fmadd_ps(y, y, Tmp);                           /* sum that with y^2 */

    __m128 RecipMag = _mm_rsqrt_ps(Sum);
    __m128 ResultVec = _mm_mul_ps(CrossProduct, RecipMag);

    v3f Result;
    _mm_store_ps(Result.Index, ResultVec);
    return Result;
}

static v3f V3f_Normalize(v3f Vec)
{
    __m128 v = _mm_load_ps(Vec.Index);                  /* [x,            y,          z,    0] */

    /* compute Mag */
    __m128 v2 = _mm_mul_ps(v, v);                       /* [x2,           y2,         z2,   0] */

    __m128 x2 = _mm_shuffle_ps(v2, v2, 0x00);           /* [x2,           x2,         x2,   x2] */
    __m128 y2 = _mm_shuffle_ps(v2, v2, 0x55);           /* [y2,           y2,         y2,   y2] */
    __m128 z2 = _mm_shuffle_ps(v2, v2, 0xAA);           /* [z2,           z2,         z2,   z2] */
    __m128 Tmp = _mm_add_ps(x2, y2);
    __m128 Sum = _mm_add_ps(Tmp, z2);                   /* [Sum,         Sum,        Sum,  Sum] */

    __m128 InvSqrt = _mm_rsqrt_ps(Sum);                 /* [Ret,         Ret,        Ret,  Ret] = 1/sqrt(Sum) */
    __m128 Normal = _mm_mul_ps(v, InvSqrt);

    v3f Ret;
    _mm_store_ps(Ret.Index, Normal);
    return Ret;
}

float V3f_DotProd(v3f A, v3f B)
{
#if 1
    return A.x*B.x + A.y*B.y + A.z*B.z;
#else
    float Result;
    __m128 VecA = _mm_load_ps(A.Index);
    __m128 VecB = _mm_load_ps(B.Index);
    __m128 ABx = _mm_mul_ss(VecA, VecB);
    __m128 Ay = _mm_shuffle_ps(VecA, VecA, 1);
    __m128 By = _mm_shuffle_ps(VecB, VecB, 1);
    __m128 Az = _mm_shuffle_ps(VecA, VecA, 2);
    __m128 Bz = _mm_shuffle_ps(VecB, VecB, 2);
    __m128 ABxy = _mm_fmadd_ps(Ay, By, ABx);
    __m128 DotProduct = _mm_fmadd_ps(Az, Bz, ABxy);
    _mm_store_ss(&Result, DotProduct);
    return Result;
#endif
}


app_state App_OnStartup(void)
{
    return (app_state) {
        .RandState = Platform_GetTimeMillisec(),

        .Light = {0, 0, -1},
        .LightMoveStart = Platform_GetTimeMillisec(),
        .LightMoveDelta = 1000.0 / 30.0,
        .LightDeltaX = 0.05,

        .LogStart = 0,
        .LogInterval = 500,
    };
}

void App_OnLoop(app_state *AppState, platform_state *PlatformState)
{
    (void)PlatformState;
    if (Platform_GetTimeMillisec() - AppState->LightMoveStart > AppState->LightMoveDelta)
    {
        AppState->Model.ColorIsValid = false;
        AppState->Light.x += AppState->LightDeltaX;
        AppState->LightMoveStart = Platform_GetTimeMillisec();
        if (!IN_RANGE(-1.0, AppState->Light.x, 1.0))
            AppState->LightDeltaX = -AppState->LightDeltaX;
        AppState->Light = V3f_Normalize(AppState->Light);
    }

    if (!AppState->Model.ColorIsValid)
    {
        obj_model *Model = &AppState->Model;

        for (u32 i = 0; i < Model->FaceCount; i++)
        {
            v3f Triangle[3];
            for (int j = 0; j < 3; j++)
            {
                Triangle[j] = Model->Vertices[Model->Faces[i].Index[j] - 1];
            }
            v3f NormalToTriangle = V3f_NormalizedCrossProd(
                V3f_Sub(Triangle[2], Triangle[0]),
                V3f_Sub(Triangle[1], Triangle[0])
            );
            //NormalToTriangle = V3f_Normalize(NormalToTriangle);

            float Intensity = V3f_DotProd(NormalToTriangle, AppState->Light);
            if (Intensity > 0)
            {
                u32 R = (u8)(Intensity * 0xFF);
                u32 G = R;
                u32 B = R;
                Model->Colors[i] = RGBColor(R, G, B);
            }
            else
            {
                Model->Colors[i] = 0;
            }
        }

        Model->ColorIsValid = true;
    }

    double LogCheckTime = Platform_GetTimeMillisec();
    if (LogCheckTime - AppState->LogStart > AppState->LogInterval)
    {
        if (AppState->Model.FaceCount)
        {
            printf("\rFrame time: %2.2fms, FPS: %2.2f; "
                    "Render time: %2.2fms; "
                    "Avg render time: %2.3fms (%gms/%g frames, %2.2f)\t\t",
                AppState->FrameTime, 1000.0 / AppState->FrameTime,
                AppState->RenderTime,
                AppState->CumulativeRenderTime / AppState->FrameCount,
                AppState->CumulativeRenderTime, AppState->FrameCount,
                AppState->FrameCount * 1000 / AppState->CumulativeRenderTime
            );
        }
        AppState->LogStart = LogCheckTime;
    }
}

void App_OnPaint(app_state *AppState, u32 *Buffer, u32 Width, u32 Height)
{
    double RenderTimeStart = Platform_GetTimeMillisec();
    AppState->FrameTime = RenderTimeStart - AppState->FrameTimeStart;
    AppState->FrameTimeStart = RenderTimeStart;

    renderer_context *RenderContext = &AppState->RenderContext;
    RenderContext->Width = Width;
    RenderContext->Height = Height;
    RenderContext->Buffer = Buffer;

    App_SetBgColor(RenderContext, RGBColor(0, 0, 0));

    obj_model *Model = &AppState->Model;
    float HalfWidth = (int)(RenderContext->Width/2);
    float HalfHeight = (int)(RenderContext->Height/2);
    float Depth = 100000000;
    __m256 ScalesHuge = _mm256_set_ps(0, Depth, HalfHeight, HalfWidth, 0, Depth, HalfHeight, HalfWidth);
    __m128 ScalesSmall = _mm256_castps256_ps128(ScalesHuge);
    __m256 OffsetsHuge = _mm256_set_ps(0, 0, HalfHeight, HalfWidth, 0, 0, HalfHeight, HalfWidth);
    __m128 OffsetsSmall = _mm256_castps256_ps128(OffsetsHuge);
    for (u32 i = 0; i < Model->FaceCount; i++)
    {
#if 0
        /* black face culling */
        if (0 == Model->Colors[i])
            continue;

        Face CurrentFace = Model->Faces[i];
        v2i Triangle[3];
        for (int j = 0; j < 3; j++)
        {
            v3f v0 = Model->Vertices[CurrentFace.Index[j] - 1];
            int x0 = (v0.x + 1.0) * RenderContext->Width * .5;
            int y0 = (v0.y + 1.0) * RenderContext->Height * .5;
            Triangle[j] = (v2i) {
                .x = x0, .y = y0,
            };
        }

        App_Draw2DTriangle(RenderContext, 
            Triangle[0], 
            Triangle[1], 
            Triangle[2], 
            Model->Colors[i]
        );
#else
        v3i TriangleVertices[4] AVX_ALIGN;
        {
#define GET_VERTEX(m, n) Model->Vertices[CurrentFace.Index[m] - 1].Index[n]
            Face CurrentFace = Model->Faces[i];
            __m256 Vertices01 = _mm256_set_ps(
                0,
                GET_VERTEX(1, 2),
                GET_VERTEX(1, 1),
                GET_VERTEX(1, 0),

                0,
                GET_VERTEX(0, 2),
                GET_VERTEX(0, 1),
                GET_VERTEX(0, 0)
            );
            __m128 Vertices2 = _mm_set_ps(
                0,
                GET_VERTEX(2, 2),
                GET_VERTEX(2, 1),
                GET_VERTEX(2, 0)
            );
#undef GET_VERTEX

            Vertices01 = _mm256_fmadd_ps(Vertices01, ScalesHuge, OffsetsHuge);
            Vertices2 = _mm_fmadd_ps(Vertices2, ScalesSmall, OffsetsSmall);
            __m256i IntVertices01 = _mm256_cvtps_epi32(Vertices01);
            __m128i IntVertices2 = _mm_cvtps_epi32(Vertices2);

            _mm256_store_si256((__m256i *)TriangleVertices, IntVertices01);
            _mm_store_si128((__m128i *)TriangleVertices + 2, IntVertices2);
        }

        App_Draw3DTriangle(RenderContext, 
            TriangleVertices[0], 
            TriangleVertices[1], 
            TriangleVertices[2], 
            Model->Colors[i]
        );
#endif 
    }

    RenderContext->Width = 0;;
    RenderContext->Height = 0;
    RenderContext->Buffer = NULL;

    AppState->RenderTime = Platform_GetTimeMillisec() - RenderTimeStart;
    if (Model->FaceCount)
    {
        AppState->CumulativeRenderTime += AppState->RenderTime;
        AppState->FrameCount += 1;
    }
}



void App_OnFileReceived(app_state *AppState, void *FileBuffer, uSize FileSize)
{
    Platform_DeallocateMemory(AppState->FileBuffer);
    AppState->FileBuffer = FileBuffer;
    AppState->FileSize = FileSize;

    App_ParseObjModel(&AppState->Model, FileBuffer, FileSize);
}

