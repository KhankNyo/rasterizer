
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
static float V3f_DotProd(v3f A, v3f B);


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
        int Residue = (((uintptr_t)Ptr + 32) & ~31) - (uintptr_t)Ptr;
        for (int i = 0; i < Residue; i++)
        {
            *Ptr++ = Color;
        }
        BufferSize -= Residue;

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

        __m256 MinVec = _mm256_set1_ps(-FLT_MAX);
        for (uSize i = 0; i < BufferSize; i += 8)
        {
            _mm256_store_ps(Ptr, MinVec);
            Ptr += 8;
        }
    }
}




typedef struct parser 
{
    const char *Ptr;
    uSize i, Len;
} parser;

static Bool8 Parser_ConsumeIfMatch(parser *Parser, char Ch)
{
    if (Parser->i < Parser->Len)
    {
        Bool8 Match = (Parser->Ptr[Parser->i] == Ch);
        Parser->i += Match;
        return Match;
    }
    return false;
}

static void Parser_SkipSpace(parser *Parser)
{
    while (Parser->i < Parser->Len)
    {
        switch (Parser->Ptr[Parser->i])
        {
        case '#': /* # python style comment */
        {
            do {
                Parser->i++;
                if (Parser->i >= Parser->Len)
                    return;
            } while (Parser->Ptr[Parser->i] != '\n');
            Parser->i++; /* skip newline */
        } break;

        case '\t': /* space */
        case '\n':
        case '\r':
        case ' ':
        {
            Parser->i++;
        } break;
        default: return;
        }
    }
}

static int Parser_ConsumeInt(parser *Parser)
{
    int Result = 0;
    Bool8 Neg = Parser_ConsumeIfMatch(Parser, '-');
    while (IN_RANGE('0', Parser->Ptr[Parser->i], '9'))
    {
        Result *= 10;
        Result += Parser->Ptr[Parser->i] - '0';
        Parser->i++;
    }

    return Neg? 
        -Result : Result;
}

static float Parser_ConsumeFloat(parser *Parser)
{
    Bool8 Neg = Parser_ConsumeIfMatch(Parser, '-');
    float Result = Parser_ConsumeInt(Parser);
    if (Parser_ConsumeIfMatch(Parser, '.'))
    {
        float Decimal = 0;
        float Powers = 1;
        while (IN_RANGE('0', Parser->Ptr[Parser->i], '9'))
        {
            Decimal *= 10;
            Powers *= 10;
            Decimal += Parser->Ptr[Parser->i] - '0';
            Parser->i++;
        }
        Result += Decimal / Powers;
    }

    if (Parser_ConsumeIfMatch(Parser, 'e') 
    || Parser_ConsumeIfMatch(Parser, 'E'))
    {
        float Exponent = Parser_ConsumeInt(Parser);
        Result *= powf(10.0f, Exponent);
    }
    return Neg? 
        -Result : Result;
}

static void Model_PushFace(obj_model *Model, const face *Face)
{
    int FaceMember = 3;
    if (Model->FaceCount >= Model->FaceCapacity/FaceMember)
    {
        uSize NewCapacity = (Model->FaceCount*2 + 1024)*FaceMember;
        uSize NewCapacityInBytes = 
            FaceMember * NewCapacity * (
                sizeof(face) 
#if defined(LTO_COMPILE_FLAG)
                + 3*(3*sizeof(float))  /* the component member itself has 3 elements, 
                                            and each 3 elements are the coords of 1 face, DO NOT refactor */
#else
                + 3*sizeof(v3f)
#endif
                + sizeof(u32)
        ) + 32 * (5); /* alignment for 5 total arrays */

        void *Ptr = Platform_AllocateMemory(NewCapacityInBytes);

        if (Model->Faces)
        {
            memcpy(Ptr, Model->Faces, Model->FaceCount * sizeof(face));
            Platform_DeallocateMemory(Model->Faces);
        }

        Model->FaceCapacity = NewCapacity;
        Model->Faces = Ptr;
    }

    Model->Faces[Model->FaceCount++] = *Face;
}

static void Model_PushVertex(obj_model *Model, const v3f *Vertex)
{
    if (Model->VertexCount >= Model->VertexCapacity)
    {
        uSize NewCapacity = Model->VertexCount*2 + 1024;
        void *Ptr = Platform_AllocateArray(v3f, NewCapacity);

        if (Model->Vertices)
        {
            memcpy(Ptr, Model->Vertices, Model->VertexCount*sizeof(v3f));
            Platform_DeallocateMemory(Model->Vertices);
        }

        Model->VertexCapacity = NewCapacity;
        Model->Vertices = Ptr;
    }

    Model->Vertices[Model->VertexCount++] = *Vertex;
}

static void App_ParseObjModel(obj_model *OutModel, const char *File, iSize FileSize)
{
    OutModel->VertexCount = 0;
    OutModel->FaceCount = 0;

    parser Parser = {
        .i = 0,
        .Len = FileSize,
        .Ptr = File,
    };
    while (Parser.i < Parser.Len)
    {
        Parser_SkipSpace(&Parser);
        if (Parser_ConsumeIfMatch(&Parser, 'v'))
        {
            if (Parser_ConsumeIfMatch(&Parser, 't')) /* 'vt' */
            {
                /* skip to newline, don't care about texture data for now */
                while (Parser.i < Parser.Len && Parser.Ptr[Parser.i] != '\n')
                    Parser.i++;
            }
            else if (Parser_ConsumeIfMatch(&Parser, 'n')) /* 'vn' */
            {
                while (Parser.i < Parser.Len && Parser.Ptr[Parser.i] != '\n')
                    Parser.i++;
            }
            else /* 'v' */
            {
                v3f v = { 0 };
                for (int i = 0; i < 3; i++)
                {
                    Parser_SkipSpace(&Parser);
                    v.Index[i] = Parser_ConsumeFloat(&Parser);
                }
                Model_PushVertex(OutModel, &v);
            }
        }
        else if (Parser_ConsumeIfMatch(&Parser, 'f'))
        {
            face Face = { 0 };
            for (int i = 0; i < 3; i++)
            {
                Parser_SkipSpace(&Parser);
                int VertexIndex = Parser_ConsumeInt(&Parser);
                int TextureIndex = -1;
                if (Parser_ConsumeIfMatch(&Parser, '/') 
                && IN_RANGE('0', Parser.Ptr[Parser.i], '9'))
                {
                    TextureIndex = Parser_ConsumeInt(&Parser);
                }
                int NormalIndex = -1;
                if (Parser_ConsumeIfMatch(&Parser, '/'))
                {
                    NormalIndex = Parser_ConsumeInt(&Parser);
                }

                Face.Index[i] = VertexIndex;
                Face.Texture[i] = TextureIndex;
                Face.Normal[i] = NormalIndex;
            }

            Model_PushFace(OutModel, &Face);
        }
        else
        {
            Parser.i++;
        }
    }

#define GET_ALIGNED_PTR(type, prevptr, scale) (type *)(((uintptr_t)(prevptr + OutModel->FaceCount*scale) + 32) & ~31)

    OutModel->Colors = GET_ALIGNED_PTR(u32, OutModel->Faces, 1);
#if defined(LTO_COMPILE_FLAG)
    OutModel->ComponentsVert = GET_ALIGNED_PTR(v3f, OutModel->Colors, 1);
#else
    OutModel->CompCount = OutModel->FaceCount*3;
    OutModel->CompX = GET_ALIGNED_PTR(float, OutModel->Colors, 1);
    OutModel->CompY = GET_ALIGNED_PTR(float, OutModel->CompX, 3);
    OutModel->CompZ = GET_ALIGNED_PTR(float, OutModel->CompY, 3);
#endif
#undef GET_ALIGNED_PTR

    OutModel->ComponentsAreValid = false;
    OutModel->ColorsAreValid = false;
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

    __m128 MagSq = _mm_dp_ps(CrossProduct, CrossProduct, 0x7F); /* [|CrossProd|^2] */
    __m128 RecipMag = _mm_rsqrt_ps(MagSq);              /* [1/|CrossProd|] */
    __m128 Result = _mm_mul_ps(CrossProduct, RecipMag);
    v3f Normal;
    _mm_store_ps(Normal.Index, Result);
    return Normal;
}

static v3f V3f_Normalize(v3f Vec)
{
    __m128 v = _mm_load_ps(Vec.Index);                  /* [x,            y,          z,        0] */

    /* compute Mag */
    __m128 MagSq = _mm_dp_ps(v, v, 0x7F);               /* [|v|^2,    |v|^2,      |v|^2,    |v|^2]*/
    __m128 RecipMag = _mm_rsqrt_ps(MagSq);              /* [1/|v|     1/|v|,      1/|v|,    1/|v|]*/
    __m128 Result = _mm_mul_ps(v, RecipMag);
    v3f Normal;
    _mm_store_ps(Normal.Index, Result);
    return Normal;
}

static v3i V3f_ToV3i(v3f v)
{
    return (v3i) {
        .x = v.x,
        .y = v.y,
        .z = v.z,
    };
}



static float V3f_DotProd(v3f A, v3f B)
{
    __m128 VecA = _mm_load_ps(A.Index);
    __m128 VecB = _mm_load_ps(B.Index);
    __m128 Result = _mm_dp_ps(VecA, VecB, 0x71); /* bitmask: does dp on first 3 elems, store in elem index 0 only */
    float Ret;
    _mm_store_ss(&Ret, Result);
    return Ret;
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
        .LogInterval = 100,
    };
}

void App_OnLoop(app_state *AppState, platform_state *PlatformState)
{
    (void)PlatformState;
    if (Platform_GetTimeMillisec() - AppState->LightMoveStart > AppState->LightMoveDelta)
    {
        AppState->Model.ColorsAreValid = false;
        AppState->Light.x += AppState->LightDeltaX;
        AppState->LightMoveStart = Platform_GetTimeMillisec();
        if (!IN_RANGE(-1.0, AppState->Light.x, 1.0))
            AppState->LightDeltaX = -AppState->LightDeltaX;
        AppState->Light = V3f_Normalize(AppState->Light);
    }

    if (!AppState->Model.ColorsAreValid)
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

        Model->ColorsAreValid = true;
    }

    if (!AppState->Model.ComponentsAreValid)
    {
        obj_model *Model = &AppState->Model;
#if defined(LTO_COMPILE_FLAG)
        for (u32 i = 0; i < Model->FaceCount; i++)
        {
            v3f v = Model->Vertices[Model->Faces[i].Index[0] - 1];
            v3f v1 = Model->Vertices[Model->Faces[i].Index[1] - 1];
            v3f v2 = Model->Vertices[Model->Faces[i].Index[2] - 1];
            Model->ComponentsVert[i*3 + 0] = v;
            Model->ComponentsVert[i*3 + 1] = v1;
            Model->ComponentsVert[i*3 + 2] = v2;
        }
#else 
        for (u32 i = 0; i < Model->FaceCount; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                Model->CompX[i*3 + j] = Model->Vertices[Model->Faces[i].Index[j] - 1].x;
                Model->CompY[i*3 + j] = Model->Vertices[Model->Faces[i].Index[j] - 1].y; 
                Model->CompZ[i*3 + j] = Model->Vertices[Model->Faces[i].Index[j] - 1].z;
            }
        }
#endif
        Model->ComponentsAreValid = true;
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
    float Depth = 10000;
#if defined(LTO_COMPILE_FLAG)
    __m256 ScalesHuge = _mm256_set_ps(0, Depth, HalfHeight, HalfWidth, 0, Depth, HalfHeight, HalfWidth);
    __m256 OffsetsHuge = _mm256_set_ps(0.5, 0.5, 0.5 + HalfHeight, 0.5 + HalfWidth, 0.5, 0.5, 0.5 + HalfHeight, 0.5 + HalfWidth);
#define FACE_PER_LOOP 2
    for (u32 i = 0; i < Model->FaceCount/FACE_PER_LOOP; i++)
    {
        v3i TriangleVertices[3*FACE_PER_LOOP] AVX_ALIGN;
        {
            float *Ptr = (float *)Model->ComponentsVert + i*(12*FACE_PER_LOOP);
            __m256 v1 = _mm256_load_ps(Ptr + 0);
            __m256 v12 = _mm256_load_ps(Ptr + 8);
            __m256 v2 = _mm256_load_ps(Ptr + 16);

            v1 = _mm256_fmadd_ps(v1, ScalesHuge, OffsetsHuge);
            v12 = _mm256_fmadd_ps(v12, ScalesHuge, OffsetsHuge);
            v2 = _mm256_fmadd_ps(v2, ScalesHuge, OffsetsHuge);

            __m256i v1i = _mm256_cvtps_epi32(v1);
            __m256i v12i = _mm256_cvtps_epi32(v12);
            __m256i v2i = _mm256_cvtps_epi32(v2);

            _mm256_store_si256((__m256i *)TriangleVertices + 0, v1i);
            _mm256_store_si256((__m256i *)TriangleVertices + 1, v12i);
            _mm256_store_si256((__m256i *)TriangleVertices + 2, v2i);
        }

#pragma GCC unroll 2
        for (int j = 0; j < FACE_PER_LOOP; j++)
        {
            App_Draw3DTriangle(RenderContext, 
                TriangleVertices[j*3 + 0], 
                TriangleVertices[j*3 + 1],
                TriangleVertices[j*3 + 2], 
                Model->Colors[i*FACE_PER_LOOP + j]
            );
        }
    }

    if (Model->FaceCount % 2)
    {
        v3i TriangleVertices[3] = {
            [0] = V3f_ToV3i(Model->ComponentsVert[Model->FaceCount - 3]),
            [1] = V3f_ToV3i(Model->ComponentsVert[Model->FaceCount - 2]),
            [2] = V3f_ToV3i(Model->ComponentsVert[Model->FaceCount - 1])
        };
        App_Draw3DTriangle(RenderContext, 
            TriangleVertices[0],
            TriangleVertices[1],
            TriangleVertices[2], 
            Model->Colors[Model->FaceCount - 1]
        );
    }
#else
    __m256 XOffset = _mm256_set1_ps(HalfWidth + .5);
    __m256 YOffset = _mm256_set1_ps(HalfHeight + .5);
    __m256 ZOffset = _mm256_set1_ps(.5);
    __m256 XScale = _mm256_set1_ps(HalfWidth);
    __m256 YScale = _mm256_set1_ps(HalfHeight);
    __m256 ZScale = _mm256_set1_ps(Depth);
    float *XPtr = Model->CompX;
    float *YPtr = Model->CompY;
    float *ZPtr = Model->CompZ;
    u32 Count = Model->FaceCount / 8;
    int Remain = Model->FaceCount % 8;
    for (u32 i = 0; i < Count; i++)
    {
        __m256 x1 = _mm256_load_ps(XPtr);
        __m256 x12 = _mm256_load_ps(XPtr + 8);
        __m256 x2 = _mm256_load_ps(XPtr + 16);

        __m256 y1 = _mm256_load_ps(YPtr);
        __m256 y12 = _mm256_load_ps(YPtr + 8);
        __m256 y2 = _mm256_load_ps(YPtr + 16);

        __m256 z1 = _mm256_load_ps(ZPtr);
        __m256 z12 = _mm256_load_ps(ZPtr + 8);
        __m256 z2 = _mm256_load_ps(ZPtr + 16);
        XPtr += 24;
        YPtr += 24;
        ZPtr += 24;

        x1 = _mm256_fmadd_ps(x1, XScale, XOffset);
        x12 = _mm256_fmadd_ps(x12, XScale, XOffset);
        x2 = _mm256_fmadd_ps(x2, XScale, XOffset);

        y1 = _mm256_fmadd_ps(y1, YScale, YOffset);
        y12 = _mm256_fmadd_ps(y12, YScale, YOffset);
        y2 = _mm256_fmadd_ps(y2, YScale, YOffset);

        z1 = _mm256_fmadd_ps(z1, ZScale, ZOffset);
        z12 = _mm256_fmadd_ps(z12, ZScale, ZOffset);
        z2 = _mm256_fmadd_ps(z2, ZScale, ZOffset);

        __m256i x1i = _mm256_cvtps_epi32(x1);
        __m256i x12i = _mm256_cvtps_epi32(x12);
        __m256i x2i = _mm256_cvtps_epi32(x2);

        __m256i y1i = _mm256_cvtps_epi32(y1);
        __m256i y12i = _mm256_cvtps_epi32(y12);
        __m256i y2i = _mm256_cvtps_epi32(y2);

        __m256i z1i = _mm256_cvtps_epi32(z1);
        __m256i z12i = _mm256_cvtps_epi32(z12);
        __m256i z2i = _mm256_cvtps_epi32(z2);

        i32 X[24] AVX_ALIGN;
        i32 Y[24] AVX_ALIGN;
        i32 Z[24] AVX_ALIGN;
        _mm256_store_si256((__m256i *)X + 0, x1i);
        _mm256_store_si256((__m256i *)X + 1, x12i);
        _mm256_store_si256((__m256i *)X + 2, x2i);

        _mm256_store_si256((__m256i *)Y + 0, y1i);
        _mm256_store_si256((__m256i *)Y + 1, y12i);
        _mm256_store_si256((__m256i *)Y + 2, y2i);

        _mm256_store_si256((__m256i *)Z + 0, z1i);
        _mm256_store_si256((__m256i *)Z + 1, z12i);
        _mm256_store_si256((__m256i *)Z + 2, z2i);
#pragma GCC unroll 8
        for (int j = 0; j < 8; j++)
        {
            v3i Vertices[3];
#pragma GCC unroll 3
            for (int k = 0; k < 3; k++)
            {
                Vertices[k].x = X[j*3 + k];
                Vertices[k].y = Y[j*3 + k];
                Vertices[k].z = Z[j*3 + k];
            }
            App_Draw3DTriangle(RenderContext, Vertices[0], Vertices[1], Vertices[2], Model->Colors[i*8 + j]);
        }
    }

    for (int i = 0; i < Remain; i++)
    {
        v3i Vertices[3];
#pragma GCC unroll 3
        for (int k = 0; k < 3; k++)
        {
            Vertices[k].x = XPtr[k];
            Vertices[k].y = YPtr[k];
            Vertices[k].z = ZPtr[k];
        }
        XPtr += 3;
        YPtr += 3;
        ZPtr += 3;
        App_Draw3DTriangle(
            RenderContext, 
            Vertices[0], 
            Vertices[1], 
            Vertices[2], 
            Model->Colors[Count*8 + i]
        );
    }
#endif


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

