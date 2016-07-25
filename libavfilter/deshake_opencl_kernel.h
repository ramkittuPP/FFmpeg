/*
 * Copyright (C) 2013 Wei Gao <weigao@multicorewareinc.com>
 * Copyright (C) 2013 Lenny Wang
 *
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVFILTER_DESHAKE_OPENCL_KERNEL_H
#define AVFILTER_DESHAKE_OPENCL_KERNEL_H

#include "libavutil/opencl.h"

const char *ff_kernel_deshake_opencl = AV_OPENCL_KERNEL(
inline unsigned char pixel(global const unsigned char *src, int x, int y,
                           int w, int h,int stride, unsigned char def)
{
    return (x < 0 || y < 0 || x >= w || y >= h) ? def : src[x + y * stride];
}

unsigned char interpolate_nearest(float x, float y, global const unsigned char *src,
                                  int width, int height, int stride, unsigned char def)
{
    return pixel(src, (int)(x + 0.5f), (int)(y + 0.5f), width, height, stride, def);
}

unsigned char interpolate_bilinear(float x, float y, global const unsigned char *src,
                                   int width, int height, int stride, unsigned char def)
{
    int x_c, x_f, y_c, y_f;
    int v1, v2, v3, v4;
    x_f = (int)x;
    y_f = (int)y;
    x_c = x_f + 1;
    y_c = y_f + 1;

    if (x_f < -1 || x_f > width || y_f < -1 || y_f > height) {
        return def;
    } else {
        v4 = pixel(src, x_f, y_f, width, height, stride, def);
        v2 = pixel(src, x_c, y_f, width, height, stride, def);
        v3 = pixel(src, x_f, y_c, width, height, stride, def);
        v1 = pixel(src, x_c, y_c, width, height, stride, def);
        return (v1*(x - x_f)*(y - y_f) + v2*((x - x_f)*(y_c - y)) +
                v3*(x_c - x)*(y - y_f) + v4*((x_c - x)*(y_c - y)));
    }
}

unsigned char interpolate_biquadratic(float x, float y, global const unsigned char *src,
                                      int width, int height, int stride, unsigned char def)
{
    int     x_c, x_f, y_c, y_f;
    unsigned char v1,  v2,  v3,  v4;
    float   f1,  f2,  f3,  f4;
    x_f = (int)x;
    y_f = (int)y;
    x_c = x_f + 1;
    y_c = y_f + 1;

    if (x_f < - 1 || x_f > width || y_f < -1 || y_f > height)
        return def;
    else {
        v4 = pixel(src, x_f, y_f, width, height, stride, def);
        v2 = pixel(src, x_c, y_f, width, height, stride, def);
        v3 = pixel(src, x_f, y_c, width, height, stride, def);
        v1 = pixel(src, x_c, y_c, width, height, stride, def);

        f1 = 1 - sqrt((x_c - x) * (y_c - y));
        f2 = 1 - sqrt((x_c - x) * (y - y_f));
        f3 = 1 - sqrt((x - x_f) * (y_c - y));
        f4 = 1 - sqrt((x - x_f) * (y - y_f));
        return (v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4) / (f1 + f2 + f3 + f4);
    }
}

inline const float clipf(float a, float amin, float amax)
{
    if      (a < amin) return amin;
    else if (a > amax) return amax;
    else               return a;
}

inline int mirror(int v, int m)
{
    while ((unsigned)v > (unsigned)m) {
        v = -v;
        if (v < 0)
            v += 2 * m;
    }
    return v;
}

kernel void avfilter_transform_luma(global unsigned char *src,
                                    global unsigned char *dst,
                                    float4 matrix,
                                    int interpolate,
                                    int fill,
                                    int src_stride_lu,
                                    int dst_stride_lu,
                                    int height,
                                    int width)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int idx_dst = y * dst_stride_lu + x;
    unsigned char def = 0;
    float x_s = x * matrix.x + y * matrix.y + matrix.z;
    float y_s = x * (-matrix.y) + y * matrix.x + matrix.w;

    if (x < width && y < height) {
        switch (fill) {
            case 0: //FILL_BLANK
                def = 0;
                break;
            case 1: //FILL_ORIGINAL
                def = src[y*src_stride_lu + x];
                break;
            case 2: //FILL_CLAMP
                y_s = clipf(y_s, 0, height - 1);
                x_s = clipf(x_s, 0, width - 1);
                def = src[(int)y_s * src_stride_lu + (int)x_s];
                break;
            case 3: //FILL_MIRROR
                y_s = mirror(y_s, height - 1);
                x_s = mirror(x_s, width - 1);
                def = src[(int)y_s * src_stride_lu + (int)x_s];
                break;
        }
        switch (interpolate) {
            case 0: //INTERPOLATE_NEAREST
                dst[idx_dst] = interpolate_nearest(x_s, y_s, src, width, height, src_stride_lu, def);
                break;
            case 1: //INTERPOLATE_BILINEAR
                dst[idx_dst] = interpolate_bilinear(x_s, y_s, src, width, height, src_stride_lu, def);
                break;
            case 2: //INTERPOLATE_BIQUADRATIC
                dst[idx_dst] = interpolate_biquadratic(x_s, y_s, src, width, height, src_stride_lu, def);
                break;
            default:
                return;
        }
    }
}

kernel void avfilter_transform_chroma(global unsigned char *src,
                                      global unsigned char *dst,
                                      float4 matrix,
                                      int interpolate,
                                      int fill,
                                      int src_stride_lu,
                                      int dst_stride_lu,
                                      int src_stride_ch,
                                      int dst_stride_ch,
                                      int height,
                                      int width,
                                      int ch,
                                      int cw)
{

    int x = get_global_id(0);
    int y = get_global_id(1);
    int pad_ch = get_global_size(1)>>1;
    global unsigned char *dst_u = dst + height * dst_stride_lu;
    global unsigned char *src_u = src + height * src_stride_lu;
    global unsigned char *dst_v = dst_u + ch * dst_stride_ch;
    global unsigned char *src_v = src_u + ch * src_stride_ch;
    src = y < pad_ch ? src_u : src_v;
    dst = y < pad_ch ? dst_u : dst_v;
    y = select(y - pad_ch, y, y < pad_ch);
    float x_s = x * matrix.x + y * matrix.y + matrix.z;
    float y_s = x * (-matrix.y) + y * matrix.x + matrix.w;
    int idx_dst = y * dst_stride_ch + x;
    unsigned char def;

    if (x < cw && y < ch) {
        switch (fill) {
            case 0: //FILL_BLANK
                def = 0;
                break;
            case 1: //FILL_ORIGINAL
                def = src[y*src_stride_ch + x];
                break;
            case 2: //FILL_CLAMP
                y_s = clipf(y_s, 0, ch - 1);
                x_s = clipf(x_s, 0, cw - 1);
                def = src[(int)y_s * src_stride_ch + (int)x_s];
                break;
            case 3: //FILL_MIRROR
                y_s = mirror(y_s, ch - 1);
                x_s = mirror(x_s, cw - 1);
                def = src[(int)y_s * src_stride_ch + (int)x_s];
                break;
        }
        switch (interpolate) {
            case 0: //INTERPOLATE_NEAREST
                dst[idx_dst] = interpolate_nearest(x_s, y_s, src, cw, ch, src_stride_ch, def);
                break;
            case 1: //INTERPOLATE_BILINEAR
                dst[idx_dst] = interpolate_bilinear(x_s, y_s, src, cw, ch, src_stride_ch, def);
                break;
            case 2: //INTERPOLATE_BIQUADRATIC
                dst[idx_dst] = interpolate_biquadratic(x_s, y_s, src, cw, ch, src_stride_ch, def);
                break;
            default:
                return;
        }
    }
}

inline void swap_xy(__local uint *lval, __local uint *rval)
{
   uint temp;
   
   if((short)*lval < (short)*rval)
   {
     temp = *lval; 
	 *lval = *rval; 
	 *rval = temp;
   }   
}

inline float block_angle(int x, int y, int cx, int cy, int mvx, int mvy)
{
	float a1, a2, diff;

	a1 = atan2((float)(y - cy), (float)(x - cx));
	a2 = atan2((float)(y - cy + mvy), (float)(x - cx + mvx));

	diff = a2 - a1;

	return (diff > GPU_M_PI) ? diff - 2 * GPU_M_PI :
		(diff < -GPU_M_PI) ? diff + 2 * GPU_M_PI :
		diff;
}

inline uint dummy_pack(float4 src)
{
	return 0;
}

inline uint dummy_sad4(uint4 src0, uint4 src1, uint src2)
{
	return 0;
}

kernel void avfilter_deshake_findmotion
    (
        global uchar4 *pref,
        global unsigned char *pcur,
        int linesize,
        global int *motion_vecs,
        volatile global int *mv_counts,
		global float *block_angles,
		global int *block_pos
    )
{
    int2 threadIdx, blockIdx;
	
    threadIdx.x = get_local_id(0);
    threadIdx.y = get_local_id(1);
	blockIdx.x  = get_group_id(0);
    blockIdx.y  = get_group_id(1);

	int offset = block_pos[blockIdx.x];
	int xPos, yPos;

	xPos = (offset >> 16);
	yPos = (offset & 0x0000FFFF);

    //pref += (linesize*((blockIdx.y*BLOCK_SIZE) + RANGE_Y) + blockIdx.x*BLOCK_SIZE + RANGE_X)/4;
    //pcur += linesize*((blockIdx.y*BLOCK_SIZE) + threadIdx.y) + blockIdx.x*BLOCK_SIZE + threadIdx.x;
    pref += (linesize* yPos + xPos)/4;
    pcur += linesize*(yPos - RANGE_Y + threadIdx.y) + xPos - RANGE_X + threadIdx.x;

	local uint4 lrefuint[BLOCK_SIZE];
	local uchar4 lref[BLOCK_SIZE][BLOCK_SIZE/4];
	if (AMD_DEVICE == 1)
	{
		global uint4 *ptmp = pref;
		if ((threadIdx.y < BLOCK_SIZE) && (threadIdx.x < 1))
			lrefuint[threadIdx.y] = ptmp[threadIdx.y*linesize / 16 + threadIdx.x];
	}
	else
	{
		if ((threadIdx.y < BLOCK_SIZE) && (threadIdx.x < (BLOCK_SIZE/4)))
			lref[threadIdx.y][threadIdx.x] = pref[threadIdx.y*linesize/4 + threadIdx.x];
	}
	
	local float lcurfloat[CURBUF_HEIGHT][CURBUF_WIDTH];
	local uchar lcur[CURBUF_HEIGHT][CURBUF_WIDTH];
	{
	   uchar num_chunks_x = CURBUF_WIDTH/16;
	   uchar num_chunks_y = CURBUF_HEIGHT/16;
	   uchar jj;

		pcur += (num_chunks_y -1) * threadIdx.y * linesize;
		\n#pragma unroll\n
		for(jj = 0; jj < num_chunks_y; jj++)
		{
			if (AMD_DEVICE == 1)
			{
				lcurfloat[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 0] = pcur[(num_chunks_x - 1) * threadIdx.x + 0];
				lcurfloat[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 1] = pcur[(num_chunks_x - 1) * threadIdx.x + 1];
				lcurfloat[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 2] = pcur[(num_chunks_x - 1) * threadIdx.x + 2];
			}
			else
			{
				lcur[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 0] = pcur[(num_chunks_x - 1) * threadIdx.x + 0];
				lcur[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 1] = pcur[(num_chunks_x - 1) * threadIdx.x + 1];
				lcur[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 2] = pcur[(num_chunks_x - 1) * threadIdx.x + 2];
			}
        
			pcur += linesize;
			mem_fence(CLK_LOCAL_MEM_FENCE);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

    uint smallest = 65280; // MAX_ERROR
    char mvx = -1; 
    char mvy = -1; 

	\n#pragma unroll\n
    for(int y=0;y<2;y++) 
	{
		\n#pragma unroll\n
		for(int x=0;x<2;x++) 
		{
			int2 curbuf_offset;
		    uint s = 0;

			curbuf_offset.x = threadIdx.x + x * 16;
			curbuf_offset.y = threadIdx.y + y * 16;

			\n#pragma unroll\n
		    for(int i=0;i<BLOCK_SIZE;i++) 
			{
				if (AMD_DEVICE == 1)
				{
					// uchar -> float -> float4 -> uint4
					float4 f1 = vload4(0, &lcurfloat[curbuf_offset.y + i][curbuf_offset.x]);
					float4 f2 = vload4(0, &lcurfloat[curbuf_offset.y + i][curbuf_offset.x + 4]);
					float4 f3 = vload4(0, &lcurfloat[curbuf_offset.y + i][curbuf_offset.x] + 8);
					float4 f4 = vload4(0, &lcurfloat[curbuf_offset.y + i][curbuf_offset.x] + 12);

					uint4 temp1;
					temp1.s0 = BYTES_PACK(f1);
					temp1.s1 = BYTES_PACK(f2);
					temp1.s2 = BYTES_PACK(f3);
					temp1.s3 = BYTES_PACK(f4);
					s += VECTOR_SAD(lrefuint[i], temp1, 0);
				}
				else
				{
					s += abs(lref[i][0].s0   - lcur[curbuf_offset.y+i][curbuf_offset.x+0]);
					s += abs(lref[i][0].s1   - lcur[curbuf_offset.y+i][curbuf_offset.x+1]);
					s += abs(lref[i][0].s2   - lcur[curbuf_offset.y+i][curbuf_offset.x+2]);
					s += abs(lref[i][0].s3   - lcur[curbuf_offset.y+i][curbuf_offset.x+3]);
					s += abs(lref[i][1].s0   - lcur[curbuf_offset.y+i][curbuf_offset.x+4]);
					s += abs(lref[i][1].s1   - lcur[curbuf_offset.y+i][curbuf_offset.x+5]);
					s += abs(lref[i][1].s2   - lcur[curbuf_offset.y+i][curbuf_offset.x+6]);
					s += abs(lref[i][1].s3   - lcur[curbuf_offset.y+i][curbuf_offset.x+7]);
					s += abs(lref[i][2].s0   - lcur[curbuf_offset.y+i][curbuf_offset.x+8]);
					s += abs(lref[i][2].s1   - lcur[curbuf_offset.y+i][curbuf_offset.x+9]);
					s += abs(lref[i][2].s2   - lcur[curbuf_offset.y+i][curbuf_offset.x+10]);
					s += abs(lref[i][2].s3   - lcur[curbuf_offset.y+i][curbuf_offset.x+11]);
					s += abs(lref[i][3].s0   - lcur[curbuf_offset.y+i][curbuf_offset.x+12]);
					s += abs(lref[i][3].s1   - lcur[curbuf_offset.y+i][curbuf_offset.x+13]);
					s += abs(lref[i][3].s2   - lcur[curbuf_offset.y+i][curbuf_offset.x+14]);
					s += abs(lref[i][3].s3   - lcur[curbuf_offset.y+i][curbuf_offset.x+15]);
				}

				mem_fence(CLK_LOCAL_MEM_FENCE);
			}
			
		    if (s < smallest) 
			{
				smallest = s;
		        mvx = RANGE_X - (char)(threadIdx.x + x*16);
		        mvy = RANGE_Y - (char)(threadIdx.y + y*16);
			}
	    } 
    } 

	if(smallest > 512)
	{
	   mvx = -1;
	   mvy = -1;
	}

	smallest |= mvx<<24;
    smallest |= (mvy<<16)&0x00ff0000;  
	
	local uint lmem[16][16];
    lmem[threadIdx.y][threadIdx.x] = smallest; 
	if(blockIdx.x < 2)
	{
       mv_counts[((16 * blockIdx.x) + threadIdx.y) * (2*MAX_R+1) + threadIdx.x] = 0;
       mv_counts[((16 * blockIdx.x) + threadIdx.y) * (2*MAX_R+1) + threadIdx.x + 16] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

    if (threadIdx.x < 8) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x+8]);
	barrier(CLK_LOCAL_MEM_FENCE);
    if ((threadIdx.x %8) < 4) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x+4]);
	barrier(CLK_LOCAL_MEM_FENCE);

    if ((threadIdx.x %4) < 2) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x+2]);
	barrier(CLK_LOCAL_MEM_FENCE);
    if ((threadIdx.x &0x1) == 0) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x+1]);
	barrier(CLK_LOCAL_MEM_FENCE);

    if (threadIdx.x == 0 || threadIdx.x == 15) 
    {
	    if (threadIdx.y < 8) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y+8][threadIdx.x]);
		barrier(CLK_LOCAL_MEM_FENCE);
	    if ((threadIdx.y %8) < 4) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y+4][threadIdx.x]);
		barrier(CLK_LOCAL_MEM_FENCE);
	    if ((threadIdx.y %4) < 2) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y+2][threadIdx.x]);
		barrier(CLK_LOCAL_MEM_FENCE);
	    if ((threadIdx.y &0x1) == 0) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y+1][threadIdx.x]);
    }
	barrier(CLK_LOCAL_MEM_FENCE);

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
		int2 gridDim ;
		char winMVx, winMVy;
		
		gridDim.x  = get_num_groups(0);
		gridDim.y  = get_num_groups(1);
		
		winMVx = (char) (lmem[15][15] >> 24);
		winMVy = (char) ((lmem[15][15] & 0x00ff0000) >> 16);

		motion_vecs[2 * blockIdx.x] = (int) winMVx;
		motion_vecs[2 * blockIdx.x + 1] = (int) winMVy;

		if ((winMVx != -1) && (winMVy != -1))
		{
			if((xPos > RANGE_X) && (yPos > RANGE_Y)) 
				block_angles[blockIdx.x] = block_angle(xPos, yPos, 0, 0, winMVx, winMVy);

			atomic_inc(&mv_counts[(winMVy + RANGE_Y) * (2*MAX_R+1) + winMVx + RANGE_X]);
		}
    }
}
/*
kernel void avfilter_deshake_findmotion_amd
    (
        global uint4 *pref,
        global unsigned char *pcur,
        int linesize,
        global int *motion_vecs,
        volatile global int *mv_counts,
        global float *block_angles,
        global int *block_pos
    )
{
	int2 threadIdx, blockIdx;

	threadIdx.x = get_local_id(0);
	threadIdx.y = get_local_id(1);
	blockIdx.x = get_group_id(0);
	blockIdx.y = get_group_id(1);

	int offset = block_pos[blockIdx.x];
	int xPos, yPos;

	xPos = (offset >> 16);
	yPos = (offset & 0x0000FFFF);

	//pref += (linesize*((blockIdx.y*BLOCK_SIZE) + RANGE_Y) + blockIdx.x*BLOCK_SIZE + RANGE_X)/4;
	//pcur += linesize*((blockIdx.y*BLOCK_SIZE) + threadIdx.y) + blockIdx.x*BLOCK_SIZE + threadIdx.x;
	pref += (linesize* yPos + xPos) / 16;
	pcur += linesize*(yPos - RANGE_Y + threadIdx.y) + xPos - RANGE_X + threadIdx.x;

	local uint4 lref[BLOCK_SIZE];
	if ((threadIdx.y < BLOCK_SIZE) && (threadIdx.x < 1))
		lref[threadIdx.y] = pref[threadIdx.y*linesize / 16 + threadIdx.x];
	
    local float lcur[CURBUF_HEIGHT][CURBUF_WIDTH];
	{
		uchar num_chunks_x = CURBUF_WIDTH / 16;
		uchar num_chunks_y = CURBUF_HEIGHT / 16;
		uchar jj;

		pcur += (num_chunks_y - 1) * threadIdx.y * linesize;
		\n#pragma unroll\n
		for (jj = 0; jj < num_chunks_y; jj++)
		{
			lcur[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 0] = pcur[(num_chunks_x - 1) * threadIdx.x + 0];
			lcur[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 1] = pcur[(num_chunks_x - 1) * threadIdx.x + 1];
			lcur[num_chunks_y * threadIdx.y + jj][num_chunks_x * threadIdx.x + 2] = pcur[(num_chunks_x - 1) * threadIdx.x + 2];

			pcur += linesize;
			mem_fence(CLK_LOCAL_MEM_FENCE);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	uint smallest = 65280; // MAX_ERROR
	char mvx = -1;
	char mvy = -1;

	\n#pragma unroll\n
	for (int y = 0; y<2; y++)
	{
		\n#pragma unroll\n
		for (int x = 0; x<2; x++)
		{
			int2 curbuf_offset;
			uint s = 0;

			curbuf_offset.x = threadIdx.x + x * 16;
			curbuf_offset.y = threadIdx.y + y * 16;

			\n#pragma unroll\n
			for (int i = 0; i<BLOCK_SIZE; i++)
			{
				// uchar -> float -> float4 -> uint4
				float4 f1 = vload4(0, &lcur[curbuf_offset.y + i][curbuf_offset.x]);
				float4 f2 = vload4(0, &lcur[curbuf_offset.y + i][curbuf_offset.x + 4]);
				float4 f3 = vload4(0, &lcur[curbuf_offset.y + i][curbuf_offset.x] + 8);
				float4 f4 = vload4(0, &lcur[curbuf_offset.y + i][curbuf_offset.x] + 12);

				uint4 temp1;
				temp1.s0 = BYTES_PACK(f1);
				temp1.s1 = BYTES_PACK(f2);
				temp1.s2 = BYTES_PACK(f3);
				temp1.s3 = BYTES_PACK(f4);
				s += VECTOR_SAD(lref[i], temp1, 0);

				mem_fence(CLK_LOCAL_MEM_FENCE);
			}

			if (s < smallest)
			{
				smallest = s;
				mvx = RANGE_X - (char)(threadIdx.x + x * 16);
				mvy = RANGE_Y - (char)(threadIdx.y + y * 16);
			}
		}
	}

	if (smallest > 512)
	{
		mvx = -1;
		mvy = -1;
	}

	smallest |= mvx << 24;
	smallest |= (mvy << 16) & 0x00ff0000;

	local uint lmem[16][16];
	lmem[threadIdx.y][threadIdx.x] = smallest;
	if (blockIdx.x < 2)
	{
		mv_counts[((16 * blockIdx.x) + threadIdx.y) * (2 * MAX_R + 1) + threadIdx.x] = 0;
		mv_counts[((16 * blockIdx.x) + threadIdx.y) * (2 * MAX_R + 1) + threadIdx.x + 16] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (threadIdx.x < 8) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x + 8]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((threadIdx.x % 8) < 4) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x + 4]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if ((threadIdx.x % 4) < 2) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x + 2]);
	barrier(CLK_LOCAL_MEM_FENCE);
	if ((threadIdx.x & 0x1) == 0) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y][threadIdx.x + 1]);
	barrier(CLK_LOCAL_MEM_FENCE);

	if (threadIdx.x == 0 || threadIdx.x == 15)
	{
		if (threadIdx.y < 8) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y + 8][threadIdx.x]);
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((threadIdx.y % 8) < 4) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y + 4][threadIdx.x]);
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((threadIdx.y % 4) < 2) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y + 2][threadIdx.x]);
		barrier(CLK_LOCAL_MEM_FENCE);
		if ((threadIdx.y & 0x1) == 0) swap_xy(&lmem[threadIdx.y][threadIdx.x], &lmem[threadIdx.y + 1][threadIdx.x]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		int2 gridDim;
		char winMVx, winMVy;

		gridDim.x = get_num_groups(0);
		gridDim.y = get_num_groups(1);

		winMVx = (char)(lmem[15][15] >> 24);
		winMVy = (char)((lmem[15][15] & 0x00ff0000) >> 16);

		motion_vecs[2 * blockIdx.x] = (int)winMVx;
		motion_vecs[2 * blockIdx.x + 1] = (int)winMVy;

		if ((winMVx != -1) && (winMVy != -1))
		{
			if ((xPos > RANGE_X) && (yPos > RANGE_Y))
				block_angles[blockIdx.x] = block_angle(xPos, yPos, 0, 0, winMVx, winMVy);

			atomic_inc(&mv_counts[(winMVy + RANGE_Y) * (2 * MAX_R + 1) + winMVx + RANGE_X]);
		}
	}
}*/
);

#endif /* AVFILTER_DESHAKE_OPENCL_KERNEL_H */
