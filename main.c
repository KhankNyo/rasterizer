
#include <windows.h>
#include "Common.h"
//#include "Renderer.c"

#define MAINTHREAD_CREATE_WINDOW (WM_USER + 0)
#define MAINTHREAD_DESTROY_WINDOW (WM_USER + 1)


/* Casey case because it's funny */
typedef struct win32_window_creation_args 
{
    DWORD ExStyle;
    DWORD Style;
    const char *ClassName;
    const char *WindowName;
    int x, y, w, h;
    HWND ParentWindow;
    HMENU Menu;
} win32_window_creation_args;

typedef struct platform_state 
{
    HWND WindowManager,
         MainWindow;

    Bool8 MouseIsDragging;
    int MouseX, MouseY;

    Bool8 KeyWasDown[0x100];
    Bool8 KeyIsDown[0x100];
    double KeyDownInit[0x100];
    unsigned char LastKeyDown;
} platform_state;

typedef struct win32_window_dimension 
{
    int x, y, w, h;
} win32_window_dimension;

typedef struct win32_paint_context 
{
    HDC Back, Front;
    HBITMAP BitmapHandle;
    int Width, Height;
    void *BitmapData;
} win32_paint_context;

typedef struct win32_buffer_data
{
    void *ViewPtr;
    uSize SizeBytes;
} win32_buffer_data;



static DWORD Win32_MainThreadID;
static double Win32_PerfCounterResolutionMillisec;
static app_state Win32_AppState;

static void Win32_Fatal(const char *ErrMsg)
{
    MessageBoxA(NULL, ErrMsg, "Fatal Error", MB_ICONERROR);
    ExitProcess(1);
}

static void Win32_SystemError(const char *Caption)
{
    DWORD ErrorCode = GetLastError();
    LPSTR ErrorText = NULL;

    FormatMessage(
        /* use system message tables to retrieve error text */
        FORMAT_MESSAGE_FROM_SYSTEM
        | FORMAT_MESSAGE_ALLOCATE_BUFFER
        | FORMAT_MESSAGE_IGNORE_INSERTS,  
        NULL,
        ErrorCode,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPSTR)&ErrorText,
        0,      /* minimum size for output buffer */
        NULL
    );
    if (NULL != ErrorText)
    {
        MessageBoxA(NULL, ErrorText, Caption, MB_ICONERROR);
        /* release memory allocated by FormatMessage() */
        LocalFree(ErrorText);
    }
}



static win32_window_dimension Win32_GetWindowDimension(HWND Window)
{
    RECT Rect;
    GetClientRect(Window, &Rect);
    return (win32_window_dimension) {
        .x = Rect.left,
        .y = Rect.top,
        .w = Rect.right - Rect.left,
        .h = Rect.bottom - Rect.top,
    };
}

static win32_paint_context Win32_BeginPaint(HWND Window)
{
    HDC FrontDC = GetDC(Window);
    HDC BackDC = CreateCompatibleDC(FrontDC);

    win32_window_dimension Dimension = Win32_GetWindowDimension(Window);
    int Width = Dimension.w,
        Height = Dimension.h;

    BITMAPINFO BitmapInfo = {
        .bmiHeader = {
            .biSize = sizeof BitmapInfo,
            .biWidth = Width,
            .biHeight = Height,
            .biPlanes = 1,
            .biBitCount = 32,
            .biCompression = BI_RGB,
        },
    };
    void *Ptr;
    HBITMAP BitmapHandle = CreateDIBSection(BackDC, &BitmapInfo, DIB_RGB_COLORS, &Ptr, NULL, 0);

    SelectObject(BackDC, BitmapHandle);
    win32_paint_context Context = {
        .Back = BackDC,
        .Front = FrontDC,
        .BitmapHandle = BitmapHandle,

        .Width = Width,
        .Height = Height,
        .BitmapData = Ptr,
    };
    return Context;
}

static void Win32_EndPaint(HWND Window, win32_paint_context *Context)
{
    BitBlt(Context->Front, 0, 0, Context->Width, Context->Height, Context->Back, 0, 0, SRCCOPY);
    DeleteObject(Context->Back);
    if (NULL != Context->BitmapHandle)
    {
        DeleteObject(Context->BitmapHandle);
    }
    ReleaseDC(Window, Context->Front);
}


static win32_buffer_data Win32_ReadFileSync(const char *FileName, DWORD dwCreationDisposition)
{
    HANDLE FileHandle = CreateFileA(FileName, GENERIC_READ, 0, NULL, dwCreationDisposition, FILE_ATTRIBUTE_NORMAL, NULL);
    if (INVALID_HANDLE_VALUE == FileHandle)
        goto CreateFileFailed;

    LARGE_INTEGER ArchaicFileSize;
    if (!GetFileSizeEx(FileHandle, &ArchaicFileSize))
        goto GetFileSizeFailed;

    uSize FileBufferSize = ArchaicFileSize.QuadPart;
    void *Buffer = Platform_AllocateMemory(FileBufferSize);
    if (NULL == Buffer)
        goto AllocateMemoryFailed;

    DWORD ReadSize;
    if (!ReadFile(FileHandle, Buffer, FileBufferSize, &ReadSize, NULL) || ReadSize != FileBufferSize)
        goto ReadFileFailed;

    CloseHandle(FileHandle);
    return (win32_buffer_data){
        .ViewPtr = Buffer,
        .SizeBytes = FileBufferSize
    };

ReadFileFailed:
    Platform_DeallocateMemory(Buffer);
AllocateMemoryFailed:
GetFileSizeFailed:
    CloseHandle(FileHandle);
CreateFileFailed:
    Win32_SystemError("Unable to open file");
    return (win32_buffer_data) { 0 };
}




static LRESULT CALLBACK Win32_ProcessMainThreadMessage(HWND Window, UINT Msg, WPARAM WParam, LPARAM LParam)
{
    /* using Casey's DTC */
    LRESULT Result = 0;
    switch (Msg)
    {
    case MAINTHREAD_CREATE_WINDOW:
    {
        win32_window_creation_args *Args = (win32_window_creation_args *)WParam;
        Result = (LRESULT)CreateWindowExA(
            Args->ExStyle, 
            Args->ClassName,
            Args->WindowName, 
            Args->Style, 
            Args->x, 
            Args->y, 
            Args->w, 
            Args->h, 
            Args->ParentWindow, 
            Args->Menu, 
            NULL, 
            NULL
        );
    } break;
    case MAINTHREAD_DESTROY_WINDOW:
    {
        HWND WindowHandle = (HWND)WParam;
        CloseWindow(WindowHandle);
    } break;
    default: Result = DefWindowProcA(Window, Msg, WParam, LParam);
    }
    return Result;
}

static LRESULT CALLBACK Win32_MainWndProc(HWND Window, UINT Msg, WPARAM WParam, LPARAM LParam)
{
    LRESULT Result = 0;
    switch (Msg)
    {
    case WM_CLOSE:
    case WM_QUIT:
    {
        /* (from msg thread) send a close msg to the main thread, 
         * it'll then send back a close window message and exit the entire process */
        PostThreadMessageA(Win32_MainThreadID, WM_CLOSE, (WPARAM)Window, 0);
    } break;
    case WM_MOUSEWHEEL:
    case WM_MOUSEMOVE:
    case WM_LBUTTONUP:
    case WM_LBUTTONDOWN:
    case WM_COMMAND:
    {
        PostThreadMessageA(Win32_MainThreadID, Msg, WParam, LParam);
    } break;

    default: Result = DefWindowProcA(Window, Msg, WParam, LParam);
    }

    return Result;
}

static Bool8 Win32_PollInputs(platform_state *State)
{
    MSG Message;
    State->KeyWasDown[State->LastKeyDown] = State->KeyIsDown[State->LastKeyDown];
    while (PeekMessageA(&Message, 0, 0, 0, PM_REMOVE))
    {
        switch (Message.message)
        {
        case WM_QUIT:
        case WM_CLOSE: 
        {
            HWND Window = (HWND)Message.wParam;
            SendMessageA(State->WindowManager, MAINTHREAD_DESTROY_WINDOW, (WPARAM)Window, 0);
            return false;
        } break;

        case WM_LBUTTONUP:
        case WM_LBUTTONDOWN:
        {
            State->MouseIsDragging = Message.message == WM_LBUTTONDOWN;
        } break;

        case WM_KEYUP:
        case WM_KEYDOWN:
        {
            Bool8 KeyIsDown = Message.message == WM_KEYDOWN;
            unsigned char Key = Message.wParam & 0xFF;
            State->LastKeyDown = Key;
            State->KeyIsDown[Key] = KeyIsDown;

            if (KeyIsDown && State->KeyDownInit[Key] == 0)
            {
                State->KeyDownInit[Key] = Platform_GetTimeMillisec();
            }
            else if (!KeyIsDown)
            {
                State->KeyDownInit[Key] = 0;
            }
        } break;

        case WM_DROPFILES:
        {
            HDROP DropInfo = (HDROP)Message.wParam;
            static char FileNameBuffer[1024];

            /* get the first dropped file's name */
            if (!DragQueryFileA(DropInfo, 0, 
                FileNameBuffer, sizeof FileNameBuffer))
            {
                Win32_SystemError("Unable to open file");
                break;
            }

            FileNameBuffer[sizeof(FileNameBuffer) - 1] = 0;
            win32_buffer_data File = Win32_ReadFileSync(FileNameBuffer, OPEN_EXISTING);
            if (File.ViewPtr)
            {
                App_OnFileReceived(&Win32_AppState, File.ViewPtr, File.SizeBytes);
            }
        } break;
        }
    }
    return true;
}

double Platform_GetTimeMillisec(void)
{
    LARGE_INTEGER Li;
    QueryPerformanceCounter(&Li);
    return Win32_PerfCounterResolutionMillisec * (double)Li.QuadPart;
}

void *Platform_AllocateMemory(uSize SizeBytes)
{
    void *Buffer = VirtualAlloc(NULL, SizeBytes, MEM_COMMIT, PAGE_READWRITE);
    if (!Buffer)
    {
        Win32_Fatal("Out of memory.");
    }
    return Buffer;
}

void Platform_DeallocateMemory(void *Ptr)
{
    VirtualFree(Ptr, 0, MEM_RELEASE);
}

Bool8 Platform_IsKeyPressed(const platform_state *State, char Key)
{
    unsigned char k = Key;
    return !State->KeyIsDown[k] && State->KeyWasDown[k];
}

Bool8 Platform_IsKeyDown(platform_state *State, char Key, double Delay)
{
    unsigned char k = Key;
    double Time = Platform_GetTimeMillisec();
    Bool8 IsDown = State->KeyIsDown[k] && Time - State->KeyDownInit[k] > Delay;
    if (IsDown)
        State->KeyDownInit[k] = Time;
    return IsDown;
}




static DWORD Win32_Main(LPVOID UserData)
{
    HWND WindowManager = UserData;
    WNDCLASSEXA WndCls = {
        .lpfnWndProc = Win32_MainWndProc,
        .lpszClassName = "Renderer",
        .cbSize = sizeof WndCls,
        .style = CS_VREDRAW | CS_HREDRAW | CS_OWNDC, 

        .hIcon = LoadIconA(NULL, IDI_APPLICATION),
        .hCursor = LoadCursorA(NULL, IDC_ARROW),
        .hInstance = GetModuleHandleA(NULL),
    };
    RegisterClassExA(&WndCls);

    win32_window_creation_args Args = {
        .x = CW_USEDEFAULT, 
        .y = CW_USEDEFAULT, 
        .w = 1080,
        .h = 720,
        .Style = WS_OVERLAPPEDWINDOW | WS_VISIBLE,
        .ExStyle = WS_EX_OVERLAPPEDWINDOW | WS_EX_ACCEPTFILES,
        .ParentWindow = NULL,
        .ClassName = WndCls.lpszClassName,
        .WindowName = "Renderer",
    };
    HWND MainWindow = (HWND)SendMessageA(WindowManager, MAINTHREAD_CREATE_WINDOW, (WPARAM)&Args, 0);
    if (NULL == MainWindow)
    {
        Win32_Fatal("Unable to create window");
    }
    ShowWindow(MainWindow, SW_SHOW);


    platform_state State = { 
        .WindowManager = WindowManager,
        .MainWindow = MainWindow,
    };

    {
        POINT Point;
        GetCursorPos(&Point);
        RECT Rect;
        GetClientRect(MainWindow, &Rect);
        State.MouseX = Point.x - Rect.left;
        State.MouseY = Point.y - Rect.top;
    }

    Win32_AppState = App_OnStartup();

    double ElapsedTime = 0;
    double LastTime = Platform_GetTimeMillisec();
    double MillisecPerFrame = 1000.0 / 60;
    while (Win32_PollInputs(&State))
    {
        App_OnLoop(&Win32_AppState, &State);
        //if (ElapsedTime > MillisecPerFrame)
        {
            win32_paint_context Context = Win32_BeginPaint(MainWindow);
            if (NULL != Context.BitmapHandle && NULL != Context.BitmapData)
            {
                App_OnPaint(&Win32_AppState, Context.BitmapData, Context.Width, Context.Height);
            }
            Win32_EndPaint(MainWindow, &Context);
        }
        double CurrentTime = Platform_GetTimeMillisec();
        ElapsedTime += CurrentTime - LastTime;
        LastTime = CurrentTime;
    }

    /* windows deallocate resources faster than us, so we let them handle that */
    (void)MainWindow;
    ExitProcess(0);
}

int WinMain(HINSTANCE Instance, HINSTANCE PrevInstance, PCHAR CmdLine, int CmdShow)
{
    (void)Instance, (void)PrevInstance, (void)CmdLine, (void)CmdShow;
    WNDCLASSEXA WndCls = {
        .lpfnWndProc = Win32_ProcessMainThreadMessage,
        .lpszClassName = "MsgHandlerCls",
        .cbSize = sizeof WndCls,
        .hIcon = LoadIconA(NULL, IDI_APPLICATION),
        .hCursor = LoadCursorA(NULL, IDC_ARROW),
        .hInstance = GetModuleHandleA(NULL),
    };
    (void)RegisterClassExA(&WndCls);

    /* this window is not visible, it's here to process the message queue, 
     * and dispatches any messages that the main thread needs */
    HWND WindowManager = CreateWindowExA(0, WndCls.lpszClassName, "Manager", 0, 0, 0, 0, 0, NULL, NULL, Instance, NULL);
    if (NULL == WindowManager)
    {
        Win32_Fatal("Unable to create window (null).");
    }

    {
        LARGE_INTEGER Li;
        QueryPerformanceFrequency(&Li);
        Win32_PerfCounterResolutionMillisec = 1000.0 / Li.QuadPart;
    }

    /* this is the main thread, the one that does actual work */
    CreateThread(NULL, 0, Win32_Main, WindowManager, 0, &Win32_MainThreadID);

    /* process the message queue */
    /* don't need any mechanism to exit the loop, 
     * because once the main thread that we've created exits, the whole program exits */
    while (1)
    {
        MSG Message;
        GetMessage(&Message, 0, 0, 0);
        TranslateMessage(&Message);

        UINT Event = Message.message;
        if (Event == WM_KEYDOWN
        || Event == WM_DROPFILES
        || Event == WM_KEYUP
        || Event == WM_LBUTTONDOWN
        || Event == WM_LBUTTONUP
        || Event == WM_SIZE
        || Event == WM_CLOSE
        || Event == WM_QUIT
        || Event == WM_MOUSEWHEEL
        || Event == WM_MOUSEMOVE)
        {
            PostThreadMessageA(Win32_MainThreadID, Event, Message.wParam, Message.lParam);
        }
        else
        {
            DispatchMessageA(&Message); 
        }
    }
    return 0;
}

