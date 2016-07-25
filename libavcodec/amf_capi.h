/*
 * C API for the AMF media library
 *
 * Copyright (c) 2015 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * disclaimer below) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma pack(8)

typedef void amfContext;
typedef void amfComponent;
typedef void amfVariant;
typedef void amfSurface;
typedef void amfData;
typedef void amfPlane;
typedef void amfPropertyStorage;        // an AMFComponent

// ****************
// *** "Platform.h"
// ****************

// -------------------------------------------------------------------------------------------------
// basic data types
// -------------------------------------------------------------------------------------------------

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

typedef     __int64 amf_int64;
typedef     __int32 amf_int32;
typedef     __int16 amf_int16;
typedef     __int8 amf_int8;

typedef     unsigned __int64 amf_uint64;
typedef     unsigned __int32 amf_uint32;
typedef     unsigned __int16 amf_uint16;
typedef     unsigned __int8 amf_uint8;
typedef     size_t amf_size;

#define AMF_STD_CALL            __stdcall
#define AMF_CDECL_CALL          __cdecl
#define AMF_FAST_CALL           __fastcall
#define AMF_INLINE              inline
#define AMF_FORCEINLINE         __forceinline

#else // !WIN32 - Linux and Mac

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef     int64_t amf_int64;
typedef     int32_t amf_int32;
typedef     int16_t amf_int16;
typedef     int8_t amf_int8;

typedef     uint64_t amf_uint64;
typedef     uint32_t amf_uint32;
typedef     uint16_t amf_uint16;
typedef     uint8_t amf_uint8;
typedef     size_t amf_size;

#define AMF_STD_CALL
#define AMF_CDECL_CALL
#define AMF_FAST_CALL
#define AMF_INLINE              __inline__
#define AMF_FORCEINLINE         __inline__

#endif // WIN32

typedef     void *amf_handle;
typedef     double amf_double;
typedef     float amf_float;

typedef     void amf_void;
typedef     int amf_bool;
typedef     long amf_long;
typedef     int amf_int;
typedef     unsigned long amf_ulong;
typedef     unsigned int amf_uint;

typedef     amf_int64 amf_pts;               // in 100 nanosecs

struct AMFRect {
    amf_int32 left;
    amf_int32 top;
    amf_int32 right;
    amf_int32 bottom;
};

// inline AMFRect AMFConstructRect(amf_int32 left, amf_int32 top, amf_int32 right, amf_int32 bottom)
// {
//    AMFRect object = {left, top, right, bottom};
//    return object;
// }

struct AMFSize {
    amf_int32 width;
    amf_int32 height;
};

// inline AMFSize AMFConstructSize(amf_int32 width, amf_int32 height)
// {
//    AMFSize object = {width, height};
//    return object;
// }

struct AMFPoint {
    amf_int32 x;
    amf_int32 y;
};

// inline AMFPoint AMFConstructPoint(amf_int32 x, amf_int32 y)
// {
//    AMFPoint object = {x, y};
//    return object;
// }

struct AMFRate {
    amf_uint32 num;
    amf_uint32 den;
};

// inline AMFRate AMFConstructRate(amf_int32 num, amf_int32 den)
// {
//    AMFRate object = {num, den};
//    return object;
// }

struct AMFRatio {
    amf_uint32 num;
    amf_uint32 den;
};

// inline AMFRatio AMFConstructRatio(amf_int32 num, amf_int32 den)
// {
//    AMFRatio object = {num, den};
//    return object;
// }

// #pragma warning(push)
// #pragma warning(disable:4201)
// #pragma pack(push, 1)
struct AMFColor {
    union {
        struct {
            amf_uint8 r;
            amf_uint8 g;
            amf_uint8 b;
            amf_uint8 a;
        };
        amf_uint32 rgba;
    };
};
// #pragma pack(pop)
// #pragma warning(pop)

// inline AMFColor AMFConstructColor(amf_uint8 r, amf_uint8 g, amf_uint8 b, amf_uint8 a)
// {
//    AMFColor object = {r, g, b, a};
//    return object;
// }

// ****************
// * Result.h
// ****************

enum AMF_RESULT {
    AMF_OK = 0,
    AMF_FAIL,

// common errors
    AMF_UNEXPECTED,

    AMF_ACCESS_DENIED,
    AMF_INVALID_ARG,
    AMF_OUT_OF_RANGE,

    AMF_OUT_OF_MEMORY,
    AMF_INVALID_POINTER,

    AMF_NO_INTERFACE,
    AMF_NOT_IMPLEMENTED,
    AMF_NOT_SUPPORTED,
    AMF_NOT_FOUND,

    AMF_ALREADY_INITIALIZED,
    AMF_NOT_INITIALIZED,

    AMF_INVALID_FORMAT,                          // invalid data format

    AMF_WRONG_STATE,
    AMF_FILE_NOT_OPEN,                           // cannot open file

// device common codes
    AMF_NO_DEVICE,

// device directx
    AMF_DIRECTX_FAILED,
// device opencl
    AMF_OPENCL_FAILED,
// device opengl
    AMF_GLX_FAILED,                              // failed to use GLX
// device XV
    AMF_XV_FAILED,                                // failed to use Xv extension
// device alsa
    AMF_ALSA_FAILED,                             // failed to use ALSA

// component common codes

    // result codes
    AMF_EOF,
    AMF_REPEAT,
    AMF_INPUT_FULL,                              // returned by AMFComponent::SubmitInput if input queue is full
    AMF_RESOLUTION_CHANGED,                      // resolution changed client needs to Drain/Terminate/Init
    AMF_RESOLUTION_UPDATED,                      // resolution changed in adaptive mode. New ROI will be set on output on newly decoded frames

    // error codes
    AMF_INVALID_DATA_TYPE,                       // invalid data type
    AMF_INVALID_RESOLUTION,                      // invalid resolution (width or height)
    AMF_CODEC_NOT_SUPPORTED,                     // codec not supported
    AMF_SURFACE_FORMAT_NOT_SUPPORTED,            // surface format not supported
    AMF_SURFACE_MUST_BE_SHARED,                  // surface should be shared (DX11: (MiscFlags & D3D11_RESOURCE_MISC_SHARED) == 0, DX9: No shared handle found)

// component video decoder
    AMF_DECODER_NOT_PRESENT,                     // failed to create the decoder
    AMF_DECODER_SURFACE_ALLOCATION_FAILED,       // failed to create the surface for decoding
    AMF_DECODER_NO_FREE_SURFACES,

// component video encoder
    AMF_ENCODER_NOT_PRESENT,                     // failed to create the encoder

// component video processor

// component video conveter

// component dem
    AMF_DEM_ERROR,
    AMF_DEM_PROPERTY_READONLY,
    AMF_DEM_REMOTE_DISPLAY_CREATE_FAILED,
    AMF_DEM_START_ENCODING_FAILED,
    AMF_DEM_QUERY_OUTPUT_FAILED,
};

// *****************
// Variant.h"
// *****************

enum AMF_VARIANT_TYPE {
    AMF_VARIANT_EMPTY = 0,

    AMF_VARIANT_BOOL   = 1,
    AMF_VARIANT_INT64  = 2,
    AMF_VARIANT_DOUBLE = 3,

    AMF_VARIANT_RECT  = 4,
    AMF_VARIANT_SIZE  = 5,
    AMF_VARIANT_POINT = 6,
    AMF_VARIANT_RATE  = 7,
    AMF_VARIANT_RATIO = 8,
    AMF_VARIANT_COLOR = 9,

    AMF_VARIANT_STRING    = 10,    // value is char*
    AMF_VARIANT_WSTRING   = 11,    // value is wchar*
    AMF_VARIANT_INTERFACE = 12,    // value is AMFInterface*
};

struct AMFVariantStruct {
    enum AMF_VARIANT_TYPE type;
    union {
        amf_bool boolValue;
        amf_int64 int64Value;
        amf_double doubleValue;
        char *stringValue;
        wchar_t *wstringValue;
        // AMFInterface*   pInterface;
        struct AMFRect rectValue;
        struct AMFSize sizeValue;
        struct AMFPoint pointValue;
        struct AMFRate rateValue;
        struct AMFRatio ratioValue;
        struct AMFColor colorValue;
    };
};

// **********
// * Data.h
// **********

enum AMF_DATA_TYPE {
    AMF_DATA_BUFFER       = 0,
    AMF_DATA_SURFACE      = 1,
    AMF_DATA_AUDIO_BUFFER = 2,
    AMF_DATA_USER         = 1000,
    // all extensions will be AMF_DATA_USER+i
};
// ----------------------------------------------------------------------------------------------
enum AMF_MEMORY_TYPE {
    AMF_MEMORY_UNKNOWN = 0,
    AMF_MEMORY_HOST    = 1,
    AMF_MEMORY_DX9     = 2,
    AMF_MEMORY_DX11    = 3,
    AMF_MEMORY_OPENCL  = 4,
    AMF_MEMORY_OPENGL  = 5,
    AMF_MEMORY_XV      = 6,
    AMF_MEMORY_GRALLOC = 7,
};

// AMF_CORE_LINK const wchar_t* const  AMF_STD_CALL AMFGetMemoryTypeName(const AMF_MEMORY_TYPE memoryType);
// AMF_CORE_LINK AMF_MEMORY_TYPE       AMF_STD_CALL AMFGetMemoryTypeByName(const wchar_t* name);

// ----------------------------------------------------------------------------------------------
enum AMF_DX_VERSION {
    AMF_DX9    = 90,
    AMF_DX9_EX = 91,
    AMF_DX11_0 = 110,
    AMF_DX11_1 = 111
};

// *******************
// * VideoEncoderVCE.h
// *******************

#define AMFVideoEncoderVCE_AVC L"AMFVideoEncoderVCE_AVC"
#define AMFVideoEncoderVCE_SVC L"AMFVideoEncoderVCE_SVC"

enum AMF_VIDEO_ENCODER_USAGE_ENUM {
    AMF_VIDEO_ENCODER_USAGE_TRANSCONDING = 0,
    AMF_VIDEO_ENCODER_USAGE_ULTRA_LOW_LATENCY,
    AMF_VIDEO_ENCODER_USAGE_LOW_LATENCY,
    AMF_VIDEO_ENCODER_USAGE_WEBCAM
};

enum AMF_VIDEO_ENCODER_PROFILE_ENUM {
    AMF_VIDEO_ENCODER_PROFILE_BASELINE = 66,
    AMF_VIDEO_ENCODER_PROFILE_MAIN     = 77,
    AMF_VIDEO_ENCODER_PROFILE_HIGH     = 100
};

enum AMF_VIDEO_ENCODER_SCANTYPE_ENUM {
    AMF_VIDEO_ENCODER_SCANTYPE_PROGRESSIVE = 0,
    AMF_VIDEO_ENCODER_SCANTYPE_INTERLACED
};

enum AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_ENUM {
    AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CONSTRAINED_QP = 0,
    AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR,
    AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_PEAK_CONSTRAINED_VBR,
    AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_LATENCY_CONSTRAINED_VBR
};

enum AMF_VIDEO_ENCODER_QUALITY_PRESET_ENUM {
    AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED = 0,
    AMF_VIDEO_ENCODER_QUALITY_PRESET_SPEED,
    AMF_VIDEO_ENCODER_QUALITY_PRESET_QUALITY
};

enum AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_ENUM {
    AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_NONE = 0,
    AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_FRAME,
    AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_TOP_FIELD,
    AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_BOTTOM_FIELD
};

enum AMF_VIDEO_ENCODER_PICTURE_TYPE_ENUM {
    AMF_VIDEO_ENCODER_PICTURE_TYPE_NONE = 0,
    AMF_VIDEO_ENCODER_PICTURE_TYPE_SKIP,
    AMF_VIDEO_ENCODER_PICTURE_TYPE_IDR,
    AMF_VIDEO_ENCODER_PICTURE_TYPE_I,
    AMF_VIDEO_ENCODER_PICTURE_TYPE_P,
    AMF_VIDEO_ENCODER_PICTURE_TYPE_B
};

enum AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_ENUM {
    AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_IDR,
    AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_I,
    AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_P,
    AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_B
};

// Static properties - can be set befor Init()

#define AMF_VIDEO_ENCODER_FRAMESIZE                             L"FrameSize"                // AMFSize; default = 0,0; Frame size
#define AMF_VIDEO_ENCODER_FRAMERATE                             L"FrameRate"                // AMFRate; default = depends on usage; Frame Rate

#define AMF_VIDEO_ENCODER_EXTRADATA                             L"ExtraData"                // AMFInterface* - > AMFBuffer*; SPS/PPS buffer - read-only
#define AMF_VIDEO_ENCODER_USAGE                                 L"Usage"                    // amf_int64(AMF_VIDEO_ENCODER_USAGE_ENUM); default = N/A; Encoder usage type. fully configures parameter set.
#define AMF_VIDEO_ENCODER_PROFILE                               L"Profile"                  // amf_int64(AMF_VIDEO_ENCODER_PROFILE_ENUM) ; default = AMF_VIDEO_ENCODER_PROFILE_MAIN;  H264 profile
#define AMF_VIDEO_ENCODER_PROFILE_LEVEL                         L"ProfileLevel"             // amf_int64; default = 42; H264 profile level
#define AMF_VIDEO_ENCODER_MAX_LTR_FRAMES                        L"MaxOfLTRFrames"           // amf_int64; default = 0; Max number of LTR frames
#define AMF_VIDEO_ENCODER_SCANTYPE                              L"ScanType"                 // amf_int64(AMF_VIDEO_ENCODER_SCANTYPE_ENUM); default = AMF_VIDEO_ENCODER_SCANTYPE_PROGRESSIVE; indicates input stream type
// Quality preset property
#define AMF_VIDEO_ENCODER_QUALITY_PRESET                        L"QualityPreset"            // amf_int64(AMF_VIDEO_ENCODER_QUALITY_PRESET_ENUM); default = depends on USAGE; Quality Preset

// dynamic properties - can be set at any time

// Rate control properties
#define AMF_VIDEO_ENCODER_B_PIC_DELTA_QP                        L"BPicturesDeltaQP"         // amf_int64; default = depends on USAGE; B-picture Delta
#define AMF_VIDEO_ENCODER_REF_B_PIC_DELTA_QP                    L"ReferenceBPicturesDeltaQP" //  amf_int64; default = depends on USAGE; Reference B-picture Delta

#define AMF_VIDEO_ENCODER_ENFORCE_HRD                           L"EnforceHRD"               // bool; default = depends on USAGE; Enforce HRD
#define AMF_VIDEO_ENCODER_FILLER_DATA_ENABLE                    L"FillerDataEnable"         // bool; default = false; Filler Data Enable

#define AMF_VIDEO_ENCODER_GOP_SIZE                              L"GOPSize"                  // amf_int64; default = 60; GOP Size, in frames
#define AMF_VIDEO_ENCODER_VBV_BUFFER_SIZE                       L"VBVBufferSize"            // amf_int64; default = depends on USAGE; VBV Buffer Size in bits
#define AMF_VIDEO_ENCODER_INITIAL_VBV_BUFFER_FULLNESS           L"InitialVBVBufferFullness" // amf_int64; default =  64; Initial VBV Buffer Fullness 0=0% 64=100%

#define AMF_VIDEO_ENCODER_MAX_AU_SIZE                           L"MaxAUSize"                // amf_int64; default = 60; Max AU Size in bits

#define AMF_VIDEO_ENCODER_MIN_QP                                L"MinQP"                    // amf_int64; default = depends on USAGE; Min QP; range = 0-51
#define AMF_VIDEO_ENCODER_MAX_QP                                L"MaxQP"                    // amf_int64; default = depends on USAGE; Max QP; range = 0-51
#define AMF_VIDEO_ENCODER_QP_I                                  L"QPI"                      // amf_int64; default = 22; I-frame QP; range = 0-51
#define AMF_VIDEO_ENCODER_QP_P                                  L"QPP"                      // amf_int64; default = 22; P-frame QP; range = 0-51
#define AMF_VIDEO_ENCODER_QP_B                                  L"QPB"                      // amf_int64; default = 22; B-frame QP; range = 0-51
#define AMF_VIDEO_ENCODER_TARGET_BITRATE                        L"TargetBitrate"            // amf_int64; default = depends on USAGE; Target bit rate in bits
#define AMF_VIDEO_ENCODER_PEAK_BITRATE                          L"PeakBitrate"              // amf_int64; default = depends on USAGE; Peak bit rate in bits
#define AMF_VIDEO_ENCODER_RATE_CONTROL_SKIP_FRAME_ENABLE        L"RateControlSkipFrameEnable"   // bool; default =  depends on USAGE; Rate Control Based Frame Skip
#define AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD                   L"RateControlMethod"        // amf_int64(AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_ENUM); default = depends on USAGE; Rate Control Method

// Picture control properties -
#define AMF_VIDEO_ENCODER_HEADER_INSERTION_SPACING              L"HeaderInsertionSpacing"   // amf_int64; default = 0; Header Insertion Spacing; range 0-1000
#define AMF_VIDEO_ENCODER_B_PIC_PATTERN                         L"BPicturesPattern"         // amf_int64; default = 3; B-picture Pattern (number of B-Frames)
#define AMF_VIDEO_ENCODER_DE_BLOCKING_FILTER                    L"DeBlockingFilter"         // bool; default = depends on USAGE; De-blocking Filter
#define AMF_VIDEO_ENCODER_B_REFERENCE_ENABLE                    L"BReferenceEnable"         // bool; default = true; Enable Refrence to B-frames
#define AMF_VIDEO_ENCODER_IDR_PERIOD                            L"IDRPeriod"                // amf_int64; default = depends on USAGE; IDR Period in frames
#define AMF_VIDEO_ENCODER_INTRA_REFRESH_NUM_MBS_PER_SLOT        L"IntraRefreshMBsNumberPerSlot" // amf_int64; default = depends on USAGE; Intra Refresh MBs Number Per Slot in Macroblocks
#define AMF_VIDEO_ENCODER_SLICES_PER_FRAME                      L"SlicesPerFrame"           // amf_int64; default = 1; Number of slices Per Frame

// Motion estimation
#define AMF_VIDEO_ENCODER_MOTION_HALF_PIXEL                     L"HalfPixel"                // bool; default= true; Half Pixel
#define AMF_VIDEO_ENCODER_MOTION_QUARTERPIXEL                   L"QuarterPixel"             // bool; default= true; Quarter Pixel

// SVC
#define AMF_VIDEO_ENCODER_NUM_TEMPORAL_ENHANCMENT_LAYERS        L"NumOfTemporalEnhancmentLayers" // amf_int64; default = 0; range = 0, min(2, caps->GetMaxNumOfTemporalLayers()) number of temporal enhancment Layers (SVC)

// Per-submission properties - can be set on input surface interface

#define AMF_VIDEO_ENCODER_END_OF_SEQUENCE                       L"EndOfSequence"            // bool; default = false; generate end of sequence
#define AMF_VIDEO_ENCODER_END_OF_STREAM                         L"EndOfStream"              // bool; default = false; generate end of stream
#define AMF_VIDEO_ENCODER_FORCE_PICTURE_TYPE                    L"ForcePictureType"         // amf_int64(AMF_VIDEO_ENCODER_PICTURE_TYPE_ENUM); default = AMF_VIDEO_ENCODER_PICTURE_TYPE_NONE; generate particular picture type
#define AMF_VIDEO_ENCODER_INSERT_AUD                            L"InsertAUD"                // bool; default = false; insert AUD
#define AMF_VIDEO_ENCODER_INSERT_SPS                            L"InsertSPS"                // bool; default = false; insert SPS
#define AMF_VIDEO_ENCODER_INSERT_PPS                            L"InsertPPS"                // bool; default = false; insert PPS
#define AMF_VIDEO_ENCODER_PICTURE_STRUCTURE                     L"PictureStructure"         // amf_int64(AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_ENUM); default = AMF_VIDEO_ENCODER_PICTURE_STRUCTURE_FRAME; indicate picture type
#define AMF_VIDEO_ENCODER_MARK_CURRENT_WITH_LTR_INDEX           L"MarkCurrentWithLTRIndex"  // //amf_int64; default = N/A; Mark current frame with LTR index
#define AMF_VIDEO_ENCODER_FORCE_LTR_REFERENCE_BITFIELD          L"ForceLTRReferenceBitfield" // amf_int64; default = 0; force LTR bit-field

// properties set by encoder on output buffer interface
#define AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE                      L"OutputDataType"           // amf_int64(AMF_VIDEO_ENCODER_OUTPUT_DATA_TYPE_ENUM); default = N/A
#define AMF_VIDEO_ENCODER_OUTPUT_MARKED_LTR_INDEX               L"MarkedLTRIndex"           // amf_int64; default = -1; Marked LTR index
#define AMF_VIDEO_ENCODER_OUTPUT_REFERENCED_LTR_INDEX_BITFIELD  L"ReferencedLTRIndexBitfield" // amf_int64; default = 0; referenced LTR bit-field

// ***********************
// * Surface.h
// ***********************

enum AMF_SURFACE_FORMAT {
    AMF_SURFACE_UNKNOWN = 0,
    AMF_SURFACE_NV12,               ///< 1 - planar Y width x height + packed UV width/2 x height/2 - 8 bit per component
    AMF_SURFACE_YV12,               ///< 2 - planar Y width x height + V width/2 x height/2 + U width/2 x height/2 - 8 bit per component
    AMF_SURFACE_BGRA,               ///< 3 - packed - 8 bit per component
    AMF_SURFACE_ARGB,               ///< 4 - packed - 8 bit per component
    AMF_SURFACE_RGBA,               ///< 5 - packed - 8 bit per component
    AMF_SURFACE_GRAY8,              ///< 6 - single component - 8 bit
    AMF_SURFACE_YUV420P,            ///< 7 - planar Y width x height + U width/2 x height/2 + V width/2 x height/2 - 8 bit per component
    AMF_SURFACE_U8V8,               ///< 8 - double component - 8 bit per component
    AMF_SURFACE_YUY2,               ///< 9 - YUY2: Byte 0=8-bit Y'0; Byte 1=8-bit Cb; Byte 2=8-bit Y'1; Byte 3=8-bit Cr

    AMF_SURFACE_FIRST = AMF_SURFACE_NV12,
    AMF_SURFACE_LAST  = AMF_SURFACE_YUY2
};

// ----------------------------------------------------------------------------------------------
// frame type
// ----------------------------------------------------------------------------------------------
enum AMF_FRAME_TYPE {
    // flags
    AMF_FRAME_STEREO_FLAG      = 0x10000000,
    AMF_FRAME_LEFT_FLAG        = AMF_FRAME_STEREO_FLAG | 0x20000000,
    AMF_FRAME_RIGHT_FLAG       = AMF_FRAME_STEREO_FLAG | 0x40000000,
    AMF_FRAME_BOTH_FLAG        = AMF_FRAME_LEFT_FLAG | AMF_FRAME_RIGHT_FLAG,
    AMF_FRAME_INTERLEAVED_FLAG = 0x01000000,
    AMF_FRAME_FIELD_FLAG       = 0x02000000,
    AMF_FRAME_EVEN_FLAG        = 0x04000000,
    AMF_FRAME_ODD_FLAG         = 0x08000000,
    // values

    AMF_FRAME_UNKNOWN = -1,

    AMF_FRAME_PROGRESSIVE = 0,

    AMF_FRAME_INTERLEAVED_EVEN_FIRST = AMF_FRAME_INTERLEAVED_FLAG | AMF_FRAME_EVEN_FLAG,
    AMF_FRAME_INTERLEAVED_ODD_FIRST  = AMF_FRAME_INTERLEAVED_FLAG | AMF_FRAME_ODD_FLAG,
    AMF_FRAME_FIELD_SINGLE_EVEN      = AMF_FRAME_FIELD_FLAG | AMF_FRAME_EVEN_FLAG,
    AMF_FRAME_FIELD_SINGLE_ODD       = AMF_FRAME_FIELD_FLAG | AMF_FRAME_ODD_FLAG,

    AMF_FRAME_STEREO_LEFT  = AMF_FRAME_LEFT_FLAG,
    AMF_FRAME_STEREO_RIGHT = AMF_FRAME_RIGHT_FLAG,
    AMF_FRAME_STEREO_BOTH  = AMF_FRAME_BOTH_FLAG,

    AMF_FRAME_INTERLEAVED_EVEN_FIRST_STEREO_LEFT = AMF_FRAME_INTERLEAVED_FLAG |
                                                   AMF_FRAME_EVEN_FLAG | AMF_FRAME_LEFT_FLAG,
    AMF_FRAME_INTERLEAVED_EVEN_FIRST_STEREO_RIGHT = AMF_FRAME_INTERLEAVED_FLAG |
                                                    AMF_FRAME_EVEN_FLAG | AMF_FRAME_RIGHT_FLAG,
    AMF_FRAME_INTERLEAVED_EVEN_FIRST_STEREO_BOTH = AMF_FRAME_INTERLEAVED_FLAG |
                                                   AMF_FRAME_EVEN_FLAG | AMF_FRAME_BOTH_FLAG,

    AMF_FRAME_INTERLEAVED_ODD_FIRST_STEREO_LEFT = AMF_FRAME_INTERLEAVED_FLAG |
                                                  AMF_FRAME_ODD_FLAG | AMF_FRAME_LEFT_FLAG,
    AMF_FRAME_INTERLEAVED_ODD_FIRST_STEREO_RIGHT = AMF_FRAME_INTERLEAVED_FLAG |
                                                   AMF_FRAME_ODD_FLAG | AMF_FRAME_RIGHT_FLAG,
    AMF_FRAME_INTERLEAVED_ODD_FIRST_STEREO_BOTH = AMF_FRAME_INTERLEAVED_FLAG |
                                                  AMF_FRAME_ODD_FLAG | AMF_FRAME_BOTH_FLAG,
};

// ***********************
// * Plane.h
// ***********************

enum AMF_PLANE_TYPE {
    AMF_PLANE_UNKNOWN = 0,
    AMF_PLANE_PACKED  = 1,            // for all packed formats: BGRA, YUY2
    AMF_PLANE_Y       = 2,
    AMF_PLANE_UV      = 3,
    AMF_PLANE_U       = 4,
    AMF_PLANE_V       = 5,
};

//
// Function Signatures
//
//
//

typedef enum AMF_RESULT (*FPAMFCREATECONTEXT)(amfContext **);
typedef enum AMF_RESULT (*FPAMFCONTEXTTERMINATE)(amfContext *);
typedef enum AMF_RESULT (*FPAMFALLOCSURFACE)(amfContext *, enum AMF_MEMORY_TYPE, enum AMF_SURFACE_FORMAT, amf_int32, amf_int32, amfSurface **);
typedef enum AMF_RESULT (*FPAMFCREATESURFACEFROMHOSTNATIVE)(amfContext *, enum AMF_SURFACE_FORMAT, amf_int32, amf_int32, amf_int32, amf_int32, void *, amfSurface **);
typedef enum AMF_RESULT (*FPAMFRELEASESURFACE)(amfSurface *);
typedef enum AMF_RESULT (*FPAMFRELEASEDATA)(amfData *);

typedef amf_size (*FPAMFBUFFERGETSIZE)(amfData *);
typedef void * (*FPAMFBUFFERGETNATIVE)(amfData *);
typedef amf_pts (*FPAMFDATAGETPTS)(amfData *);
typedef void (*FPAMFDATASETPTS)(amfData *, amf_pts);
typedef amf_pts (*FPAMFDATAGETDURATION)(amfData *);
typedef void (*FPAMFDATASETDURATION)(amfData *, amf_pts);

typedef enum AMF_SURFACE_FORMAT (*FPAMFSURFACEGETFORMAT)(amfSurface *);
typedef amf_size (*FPAMFSURFACEGETPLANESCOUNT)(amfSurface *);
typedef amfPlane * (*FPAMFSURFACEGETPLANEAT)(amfSurface *, amf_size);
typedef amfPlane * (*FPAMFSURFACEGETPLANE)(amfSurface *, enum AMF_PLANE_TYPE);

typedef enum AMF_PLANE_TYPE (*FPAMFPLANEGETTYPE)(amfPlane *);
typedef void * (*FPAMFPLANEGETNATIVE)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETPIXELSIZEINBYTES)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETOFFSETX)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETOFFSETY)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETWIDTH)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETHEIGHT)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETHPITCH)(amfPlane *);
typedef amf_int32 (*FPAMFPLANEGETVPITCH)(amfPlane *);

typedef enum AMF_RESULT (*FPAMFCREATECOMPONENT)(amfContext *, const wchar_t *, amfComponent **);
typedef enum AMF_RESULT (*FPAMFCOMPONENTINIT)(amfComponent *, enum AMF_SURFACE_FORMAT, amf_int32, amf_int32);
typedef enum AMF_RESULT (*FPAMFCOMPONENTREINIT)(amfComponent *, amf_int32, amf_int32);
typedef enum AMF_RESULT (*FPAMFCOMPONENTTERMINATE)(amfComponent *);
typedef enum AMF_RESULT (*FPAMFCOMPONENTDRAIN)(amfComponent *);
typedef enum AMF_RESULT (*FPAMFCOMPONENTFLUSH)(amfComponent *);
typedef enum AMF_RESULT (*FPAMFCOMPONENTSUBMITINPUT)(amfComponent *, amfSurface *);
typedef enum AMF_RESULT (*FPAMFCOMPONENTQUERYOUTPUT)(amfComponent *, amfData **);

typedef enum AMF_RESULT (*FPAMFSETPROPERTYBOOL)(amfPropertyStorage *, const wchar_t *, amf_bool);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYINT64)(amfPropertyStorage *, const wchar_t *, amf_int64);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYDOUBLE)(amfPropertyStorage *, const wchar_t *, amf_double);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYSTRING)(amfPropertyStorage *, const wchar_t *, const char *);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYWSTRING)(amfPropertyStorage *, const wchar_t *, const wchar_t *);
// typedef enum AMF_RESULT (*FPAMFSETPROPERTYINTERFACE)(amfPropertyStorage *, const wchar_t*, amfInterface*);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYRECT)(amfPropertyStorage *, const wchar_t *, const struct AMFRect *);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYSIZE)(amfPropertyStorage *, const wchar_t *, const struct AMFSize *);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYPOINT)(amfPropertyStorage *, const wchar_t *, const struct AMFPoint *);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYRATE)(amfPropertyStorage *, const wchar_t *, const struct AMFRate *);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYRATIO)(amfPropertyStorage *, const wchar_t *, const struct AMFRatio *);
typedef enum AMF_RESULT (*FPAMFSETPROPERTYCOLOR)(amfPropertyStorage *, const wchar_t *, const struct AMFColor *);

typedef enum AMF_RESULT (*FPAMFGETPROPERTYBOOL)(amfPropertyStorage *, const wchar_t *, amf_bool *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYINT64)(amfPropertyStorage *, const wchar_t *, amf_int64 *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYDOUBLE)(amfPropertyStorage *, const wchar_t *, amf_double *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYSTRING)(amfPropertyStorage *, const wchar_t *, const char **);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYWSTRING)(amfPropertyStorage *, const wchar_t *, const wchar_t **);
// typedef enum AMF_RESULT (*FPAMFGETPROPERTYINTERFACE)(amfPropertyStorage *, const wchar_t*, amfInterface**);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYRECT)(amfPropertyStorage *, const wchar_t *, struct AMFRect *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYSIZE)(amfPropertyStorage *, const wchar_t *, struct AMFSize *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYPOINT)(amfPropertyStorage *, const wchar_t *, struct AMFPoint *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYRATE)(amfPropertyStorage *, const wchar_t *, struct AMFRate *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYRATIO)(amfPropertyStorage *, const wchar_t *, struct AMFRatio *);
typedef enum AMF_RESULT (*FPAMFGETPROPERTYCOLOR)(amfPropertyStorage *, const wchar_t *, struct AMFColor *);

typedef enum AMF_RESULT (*FPAMFINITENCODER)(amfComponent *, enum AMF_SURFACE_FORMAT, amf_int32, amf_int32);
typedef enum AMF_RESULT (*FPAMFCOMPONENTGETEXTRADATA)(amfComponent *, amfData **);
typedef enum AMF_RESULT (*FPAMFCOPYYUV420HOSTTONV12DX9)(unsigned char **, amf_int32 *, amfSurface *);

//
// Function Access
//
//
//

extern FPAMFCREATECONTEXT amfCreateContext;
extern FPAMFCONTEXTTERMINATE amfContextTerminate;
extern FPAMFALLOCSURFACE amfAllocSurface;
extern FPAMFCREATESURFACEFROMHOSTNATIVE amfCreateSurfaceFromHostNative;
extern FPAMFRELEASESURFACE amfReleaseSurface;
extern FPAMFRELEASEDATA amfReleaseData;

extern FPAMFBUFFERGETSIZE amfBufferGetSize;
extern FPAMFBUFFERGETNATIVE amfBufferGetNative;
extern FPAMFDATAGETPTS amfDataGetPts;
extern FPAMFDATASETPTS amfDataSetPts;
extern FPAMFDATAGETDURATION amfDataGetDuration;
extern FPAMFDATASETDURATION amfDataSetDuration;

extern FPAMFSURFACEGETFORMAT amfSurfaceGetFormat;
extern FPAMFSURFACEGETPLANESCOUNT amfSurfaceGetPlanesCount;
extern FPAMFSURFACEGETPLANEAT amfSurfaceGetPlaneAt;
extern FPAMFSURFACEGETPLANE amfSurfaceGetPlane;

extern FPAMFPLANEGETTYPE amfPlaneGetType;
extern FPAMFPLANEGETNATIVE amfPlaneGetNative;
extern FPAMFPLANEGETPIXELSIZEINBYTES amfPlaneGetSizeInBytes;
extern FPAMFPLANEGETOFFSETX amfPlaneGetOffsetX;
extern FPAMFPLANEGETOFFSETY amfPlaneGetOffsetY;
extern FPAMFPLANEGETWIDTH amfPlaneGetWidth;
extern FPAMFPLANEGETHEIGHT amfPlaneGetHeight;
extern FPAMFPLANEGETHPITCH amfPlaneGetHPitch;
extern FPAMFPLANEGETVPITCH amfPlaneGetVPitch;

extern FPAMFCREATECOMPONENT amfCreateComponent;
extern FPAMFCOMPONENTINIT amfComponentInit;
extern FPAMFCOMPONENTREINIT amfComponentReInit;
extern FPAMFCOMPONENTTERMINATE amfComponentTerminate;
extern FPAMFCOMPONENTDRAIN amfComponentDrain;
extern FPAMFCOMPONENTFLUSH amfComponentFlush;
extern FPAMFCOMPONENTSUBMITINPUT amfComponentSubmitInput;
extern FPAMFCOMPONENTQUERYOUTPUT amfComponentQueryOutput;

extern FPAMFSETPROPERTYBOOL amfSetPropertyBool;
extern FPAMFSETPROPERTYINT64 amfSetPropertyInt64;
extern FPAMFSETPROPERTYDOUBLE amfSetPropertyDouble;
extern FPAMFSETPROPERTYSTRING amfSetPropertyString;
extern FPAMFSETPROPERTYWSTRING amfSetPropertyWString;
// extern PFNAMFSETPROPERTYINTERFACE amfSetPropertyInterface;
extern FPAMFSETPROPERTYRECT amfSetPropertyRect;
extern FPAMFSETPROPERTYSIZE amfSetPropertySize;
extern FPAMFSETPROPERTYPOINT amfSetPropertyPoint;
extern FPAMFSETPROPERTYRATE amfSetPropertyRate;
extern FPAMFSETPROPERTYRATIO amfSetPropertyRatio;
extern FPAMFSETPROPERTYCOLOR amfSetPropertyColor;

extern FPAMFGETPROPERTYBOOL amfGetPropertyBool;
extern FPAMFGETPROPERTYINT64 amfGetPropertyInt64;
extern FPAMFGETPROPERTYDOUBLE amfGetPropertyDouble;
extern FPAMFGETPROPERTYSTRING amfGetPropertyString;
extern FPAMFGETPROPERTYWSTRING amfGetPropertyWString;
// extern PFNAMFGETPROPERTYINTERFACE amfSetPropertyInterface;
extern FPAMFGETPROPERTYRECT amfGetPropertyRect;
extern FPAMFGETPROPERTYSIZE amfGetPropertySize;
extern FPAMFGETPROPERTYPOINT amfGetPropertyPoint;
extern FPAMFGETPROPERTYRATE amfGetPropertyRate;
extern FPAMFGETPROPERTYRATIO amfGetPropertyRatio;
extern FPAMFGETPROPERTYCOLOR amfGetPropertyColor;

extern FPAMFINITENCODER amfInitEncoder;
extern FPAMFCOMPONENTGETEXTRADATA amfComponentGetExtraData;
extern FPAMFCOPYYUV420HOSTTONV12DX9 amfCopyYUV420HostToNV12DX9;

enum AMF_RESULT amf_capi_init(void);
void amf_capi_exit(void);