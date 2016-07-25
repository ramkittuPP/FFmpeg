/*
 * Copyright (C) 2013 Wei Gao <weigao@multicorewareinc.com>
 * Copyright (C) 2013 Lenny Wang
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

/**
 * @file
 * transform input video
 */

#include "libavutil/common.h"
#include "libavutil/dict.h"
#include "libavutil/pixdesc.h"
#include "deshake_opencl.h"
#include "libavutil/opencl_internal.h"

#define PLANE_NUM 3
#define ROUND_TO_16(a) (((((a) - 1)/16)+1)*16)

#ifdef OCL_FINDMOTION	

static int cmp_ocl(const float *a, const float *b)
{
    return *a < *b ? -1 : ( *a > *b ? 1 : 0 );
}

/**
 * Cleaned mean (cuts off 20% of values to remove outliers and then averages)
 */
static float clean_mean_ocl(float *values, int count)
{
    float mean = 0;
    int cut = count / 5;
    int x;

    qsort(values, count, sizeof(float), (void*)cmp_ocl);

    for (x = cut; x < count - cut; x++) {
        mean += values[x];
    }

    return mean / (count - cut * 2);
}

/**
 * Find the contrast of a given block. When searching for global motion we
 * really only care about the high contrast blocks, so using this method we
 * can actually skip blocks we don't care much about.
 */
static int block_contrast_gpu(uint8_t *src, int x, int y, int stride, int blocksize)
{
    int highest = 0;
    int lowest = 255;
    int i, j, pos;

    for (i = 0; i <= blocksize * 2; i++) {
        // We use a width of 16 here to match the sad function
        for (j = 0; j <= 15; j++) {
            pos = (y - i) * stride + (x - j);
            if (src[pos] < lowest)
                lowest = src[pos];
            else if (src[pos] > highest) {
                highest = src[pos];
            }
        }
    }

    return highest - lowest;
}

/**
* Count valid blocks for which motion estimation gets carried out
*/
static int valid_blocks(DeshakeContext *deshake, uint8_t *src, int stride, int width, int height, int *inOffsetTable)
{
	int x, y;
	int contrast, blocks_count = 0;

	for (y = deshake->ry; y < height - deshake->ry - (deshake->blocksize * 2); y += deshake->blocksize * 2)
	{
		for (x = deshake->rx; x < width - deshake->rx - 16; x += 16)
		{
			contrast = block_contrast_gpu(src, x, y, stride, deshake->blocksize);
			if (contrast > deshake->contrast)
			{
				inOffsetTable[blocks_count] = (x << 16) | y;
				blocks_count++;
			}
		}
	}

	return blocks_count;
}

/**
* Top level opencl function
*/
int ff_opencl_findmotion(AVFilterContext *ctx, uint8_t *src1, uint8_t *src2,
                        int width, int height, int stride, Transform *t)
{
    int ret = 0;
    cl_int status;
    DeshakeContext *deshake = ctx->priv;
    FFOpenclParam param_fm = {0};
    param_fm.ctx = ctx;
    param_fm.kernel = deshake->opencl_ctx.kernel_findmotion;
    size_t cl_mv_outsize, cl_blkangles_outsize;
    size_t cl_mv_countsize, cl_blkoffset_tablesize;

    size_t local_worksize[2] = {16, 16};
	int  globalDim[2] = {(width - 2 * deshake->rx - 16)/16, (height - 2 * deshake->ry - 1 ) / (2 * deshake->blocksize)};
    size_t global_worksize_fm[2] = { globalDim[0] * local_worksize[0], globalDim[1] * local_worksize[1] };

    cl_blkoffset_tablesize = (width * height) * sizeof(int) / 256;

    if(!deshake->opencl_offsetTable)
	{
		deshake->opencl_offsetTable = (int*)malloc(cl_blkoffset_tablesize);
	}
	
	int validBlks = 0;
	//memset((void *)deshake->opencl_offsetTable, 0, cl_blkoffset_tablesize);
	validBlks = valid_blocks(deshake, src2, stride, width, height, deshake->opencl_offsetTable);
	
	cl_mv_outsize = validBlks * sizeof(IntMotionVector);
	cl_mv_countsize = (2 * MAX_R + 1) * (2 * MAX_R + 1) * sizeof(int);
	cl_blkangles_outsize = validBlks * sizeof(float); 

    if(src1 == src2)
    {
      	deshake->opencl_ctx.cl_refbuf = deshake->opencl_ctx.cl_inbuf;
    }
	
    if ((!deshake->opencl_ctx.cl_refbuf) || (!deshake->opencl_ctx.cl_mvbuf) ||
		(!deshake->opencl_ctx.cl_mvcountbuf) || (!deshake->opencl_ctx.cl_blockanglesbuf)) {

		if (!deshake->opencl_ctx.cl_tempbuf) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_tempbuf,
                                            deshake->opencl_ctx.cl_refbuf_size,
											CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, NULL); //
            if (ret < 0)
                return ret;
        }
		
        if (!deshake->opencl_ctx.cl_mvbuf) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_mvbuf,
                                            cl_mv_outsize,
                                            CL_MEM_READ_WRITE, NULL);
            if (ret < 0)
                return ret;
        }
        if (!deshake->opencl_ctx.cl_mvcountbuf) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_mvcountbuf,
                                            cl_mv_countsize,
											CL_MEM_READ_WRITE, NULL); //  | CL_MEM_ALLOC_HOST_PTR
            if (ret < 0)
                return ret;

			memset((void *)&deshake->counts[0][0], 0, cl_mv_countsize);
			av_opencl_buffer_write(deshake->opencl_ctx.cl_mvcountbuf, &deshake->counts[0][0], cl_mv_countsize);
        }
        if (!deshake->opencl_ctx.cl_blockanglesbuf) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_blockanglesbuf,
                                            cl_blkangles_outsize,
											CL_MEM_READ_WRITE, NULL); //  | CL_MEM_ALLOC_HOST_PTR
            if (ret < 0)
                return ret;
        }
        if (!deshake->opencl_ctx.cl_blockPosTable) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_blockPosTable,
                                            cl_blkoffset_tablesize,
											CL_MEM_READ_ONLY, NULL); //  | CL_MEM_ALLOC_HOST_PTR
            if (ret < 0)
                return ret;
        }
    }

	
    ret = avpriv_opencl_set_parameter(&param_fm,
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_refbuf),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_inbuf),
                                  FF_OPENCL_PARAM_INFO(stride),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_mvbuf),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_mvcountbuf),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_blockanglesbuf),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_blockPosTable),
                                  NULL);
    if (ret < 0)
        return ret;
	
	av_opencl_buffer_write(deshake->opencl_ctx.cl_blockPosTable, deshake->opencl_offsetTable, cl_blkoffset_tablesize);

	size_t local_work[2] = {16, 16 };
	size_t global_work[2] = { validBlks * local_work[0], local_work[1]};
	
    status = clEnqueueNDRangeKernel(deshake->opencl_ctx.command_queue,
                                    deshake->opencl_ctx.kernel_findmotion, 2, NULL,
                                    global_work, local_work, 0, NULL, NULL);
	clFinish(deshake->opencl_ctx.command_queue);

    if (status != CL_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "OpenCL run kernel error occurred: %s\n", av_opencl_errstr(status));
        return AVERROR_EXTERNAL;
    }

	if(!deshake->angles_ocl)
	{
		deshake->angles_ocl = (float*)av_mallocz(cl_blkangles_outsize);
		
	}
	
    ret = av_opencl_buffer_read(&deshake->counts[0][0], deshake->opencl_ctx.cl_mvcountbuf, cl_mv_countsize);
    ret |= av_opencl_buffer_read(deshake->angles_ocl, deshake->opencl_ctx.cl_blockanglesbuf, cl_blkangles_outsize);
	
    if (ret < 0)
        return ret;

	
    int center_x = 0, center_y = 0;
	int ii, jj, temp, pos = 0;
#define RANGE_X 16
#define RANGE_Y 16

	for (ii = 0; ii < (2 * RANGE_Y); ii++)
	{
		for (jj = 0; jj < (2 * RANGE_X); jj++)
		{
			if (deshake->counts[ii][jj] > 0)
			{
				center_y += (ii - RANGE_Y) * deshake->counts[ii][jj];
				center_x += (jj - RANGE_X) * deshake->counts[ii][jj];
			
				pos += deshake->counts[ii][jj];
			}
		}
	}
	
    if (pos) {
         center_x /= pos;
         center_y /= pos;
         t->angle = clean_mean_ocl(deshake->angles_ocl, pos);
         if (t->angle < 0.001)
              t->angle = 0;
    } else {
         t->angle = 0;
    }

    int x, y;
    float p_x, p_y;
    int count_max_value = 0;
    // Find the most common motion vector in the frame and use it as the gmv
    for (y = deshake->ry * 2; y >= 0; y--) {
        for (x = 0; x < deshake->rx * 2 + 1; x++) {
            //av_log(NULL, AV_LOG_ERROR, "%5d ", deshake->counts[x][y]);
            if (deshake->counts[x][y] > count_max_value) {
                t->vec.x = x - deshake->rx;
                t->vec.y = y - deshake->ry;
                count_max_value = deshake->counts[x][y];
            }
        }
        //av_log(NULL, AV_LOG_ERROR, "\n");
    }

    p_x = (center_x - width / 2.0);
    p_y = (center_y - height / 2.0);
    t->vec.x += (cos(t->angle)-1)*p_x  - sin(t->angle)*p_y;
    t->vec.y += sin(t->angle)*p_x  + (cos(t->angle)-1)*p_y;

    // Clamp max shift & rotation?
    t->vec.x = av_clipf(t->vec.x, -deshake->rx * 2, deshake->rx * 2);
    t->vec.y = av_clipf(t->vec.y, -deshake->ry * 2, deshake->ry * 2);
    t->angle = av_clipf(t->angle, -0.1, 0.1);

	
    return ret;
}

#endif
int ff_opencl_transform(AVFilterContext *ctx,
                        int width, int height, int cw, int ch,
                        const float *matrix_y, const float *matrix_uv,
                        enum InterpolateMethod interpolate,
                        enum FillMethod fill, AVFrame *in, AVFrame *out)
{
    int ret = 0;
    cl_int status;
    DeshakeContext *deshake = ctx->priv;
    float4 packed_matrix_lu = {matrix_y[0], matrix_y[1], matrix_y[2], matrix_y[5]};
    float4 packed_matrix_ch = {matrix_uv[0], matrix_uv[1], matrix_uv[2], matrix_uv[5]};
    size_t global_worksize_lu[2] = {(size_t)ROUND_TO_16(width), (size_t)ROUND_TO_16(height)};
    size_t global_worksize_ch[2] = {(size_t)ROUND_TO_16(cw), (size_t)(2*ROUND_TO_16(ch))};
    size_t local_worksize[2] = {16, 16};
    FFOpenclParam param_lu = {0};
    FFOpenclParam param_ch = {0};
    param_lu.ctx = param_ch.ctx = ctx;
    param_lu.kernel = deshake->opencl_ctx.kernel_luma;
    param_ch.kernel = deshake->opencl_ctx.kernel_chroma;

    if ((unsigned int)interpolate > INTERPOLATE_BIQUADRATIC) {
        av_log(ctx, AV_LOG_ERROR, "Selected interpolate method is invalid\n");
        return AVERROR(EINVAL);
    }
    ret = avpriv_opencl_set_parameter(&param_lu,
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_inbuf),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_outbuf),
                                  FF_OPENCL_PARAM_INFO(packed_matrix_lu),
                                  FF_OPENCL_PARAM_INFO(interpolate),
                                  FF_OPENCL_PARAM_INFO(fill),
                                  FF_OPENCL_PARAM_INFO(in->linesize[0]),
                                  FF_OPENCL_PARAM_INFO(out->linesize[0]),
                                  FF_OPENCL_PARAM_INFO(height),
                                  FF_OPENCL_PARAM_INFO(width),
                                  NULL);
    if (ret < 0)
        return ret;
    ret = avpriv_opencl_set_parameter(&param_ch,
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_inbuf),
                                  FF_OPENCL_PARAM_INFO(deshake->opencl_ctx.cl_outbuf),
                                  FF_OPENCL_PARAM_INFO(packed_matrix_ch),
                                  FF_OPENCL_PARAM_INFO(interpolate),
                                  FF_OPENCL_PARAM_INFO(fill),
                                  FF_OPENCL_PARAM_INFO(in->linesize[0]),
                                  FF_OPENCL_PARAM_INFO(out->linesize[0]),
                                  FF_OPENCL_PARAM_INFO(in->linesize[1]),
                                  FF_OPENCL_PARAM_INFO(out->linesize[1]),
                                  FF_OPENCL_PARAM_INFO(height),
                                  FF_OPENCL_PARAM_INFO(width),
                                  FF_OPENCL_PARAM_INFO(ch),
                                  FF_OPENCL_PARAM_INFO(cw),
                                  NULL);
    if (ret < 0)
        return ret;
    status = clEnqueueNDRangeKernel(deshake->opencl_ctx.command_queue,
                                    deshake->opencl_ctx.kernel_luma, 2, NULL,
                                    global_worksize_lu, local_worksize, 0, NULL, NULL);
    status |= clEnqueueNDRangeKernel(deshake->opencl_ctx.command_queue,
                                    deshake->opencl_ctx.kernel_chroma, 2, NULL,
                                    global_worksize_ch, local_worksize, 0, NULL, NULL);
    if (status != CL_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "OpenCL run kernel error occurred: %s\n", av_opencl_errstr(status));
        return AVERROR_EXTERNAL;
    }
    ret = av_opencl_buffer_read_image(out->data, deshake->opencl_ctx.out_plane_size,
                                      deshake->opencl_ctx.plane_num, deshake->opencl_ctx.cl_outbuf,
                                      deshake->opencl_ctx.cl_outbuf_size);
    if (ret < 0)
        return ret;

#ifdef OCL_FINDMOTION	
	deshake->opencl_ctx.cl_refbuf = deshake->opencl_ctx.cl_inbuf;
	deshake->opencl_ctx.cl_inbuf = deshake->opencl_ctx.cl_tempbuf;
	deshake->opencl_ctx.cl_tempbuf = deshake->opencl_ctx.cl_refbuf;
#endif	
    return ret;
}

int ff_opencl_deshake_init(AVFilterContext *ctx)
{
    int ret = 0;
    DeshakeContext *deshake = ctx->priv;
    ret = av_opencl_init(NULL);
    if (ret < 0)
        return ret;
    deshake->opencl_ctx.plane_num = PLANE_NUM;
    deshake->opencl_ctx.command_queue = av_opencl_get_command_queue();
    if (!deshake->opencl_ctx.command_queue) {
        av_log(ctx, AV_LOG_ERROR, "Unable to get OpenCL command queue in filter 'deshake'\n");
        return AVERROR(EINVAL);
    }

	int AMD_device = 0;
	int cb = 0;
	cl_device_id device_id = av_opencl_get_device_id();
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, 0, NULL, &cb);
    char *vendor = (char *)malloc(cb);
	clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, cb, vendor, 0);
	if (strstr(vendor, "Advanced Micro Devices") || strstr(vendor, "AMD"))
		AMD_device = 1;
	free(vendor);
	
    char *flags = (char *)malloc(256);
	if (AMD_device == 1)
        sprintf(flags, "-D AMD_DEVICE=1 -D BYTES_PACK(a)=amd_pack(a) -D VECTOR_SAD(x,y,z)=amd_sad4(x,y,z) -D RANGE_X=16 -D RANGE_Y=16 -D BLOCK_SIZE=16 -D MAX_R=64 -D GPU_M_PI=3.14159265358979323846 -D CURBUF_WIDTH=48 -D CURBUF_HEIGHT=48");
	else	
        sprintf(flags, "-D AMD_DEVICE=0 -D BYTES_PACK(a)=dummy_pack(a) -D VECTOR_SAD(x,y,z)=dummy_sad4(x,y,z) -D RANGE_X=16 -D RANGE_Y=16 -D BLOCK_SIZE=16 -D MAX_R=64 -D GPU_M_PI=3.14159265358979323846 -D CURBUF_WIDTH=48 -D CURBUF_HEIGHT=48");
    deshake->opencl_ctx.program = av_opencl_compile("avfilter_transform", flags);
    if (!deshake->opencl_ctx.program) {
        av_log(ctx, AV_LOG_ERROR, "OpenCL failed to compile program 'avfilter_transform'\n");
        return AVERROR(EINVAL);
    }
    free(flags);	
	
#ifdef OCL_FINDMOTION
    if (!deshake->opencl_ctx.kernel_findmotion) {
	//	if (AMD_device == 1)
    //        deshake->opencl_ctx.kernel_findmotion = clCreateKernel(deshake->opencl_ctx.program,
    //                                                     "avfilter_deshake_findmotion_amd", &ret);
	//	else
            deshake->opencl_ctx.kernel_findmotion = clCreateKernel(deshake->opencl_ctx.program,
                                                         "avfilter_deshake_findmotion", &ret);
        if (ret != CL_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "OpenCL failed to create kernel 'avfilter_deshake_findmotion'\n");
            return AVERROR(EINVAL);
        }
    }
#endif	
    if (!deshake->opencl_ctx.kernel_luma) {
        deshake->opencl_ctx.kernel_luma = clCreateKernel(deshake->opencl_ctx.program,
                                                         "avfilter_transform_luma", &ret);
        if (ret != CL_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "OpenCL failed to create kernel 'avfilter_transform_luma'\n");
            return AVERROR(EINVAL);
        }
    }
    if (!deshake->opencl_ctx.kernel_chroma) {
        deshake->opencl_ctx.kernel_chroma = clCreateKernel(deshake->opencl_ctx.program,
                                                           "avfilter_transform_chroma", &ret);
        if (ret != CL_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "OpenCL failed to create kernel 'avfilter_transform_chroma'\n");
            return AVERROR(EINVAL);
        }
    }
    return ret;
}

void ff_opencl_deshake_uninit(AVFilterContext *ctx)
{
    DeshakeContext *deshake = ctx->priv;
    av_opencl_buffer_release(&deshake->opencl_ctx.cl_inbuf);
    av_opencl_buffer_release(&deshake->opencl_ctx.cl_outbuf);
#ifdef OCL_FINDMOTION	
    clReleaseKernel(deshake->opencl_ctx.kernel_findmotion);
#endif	
    clReleaseKernel(deshake->opencl_ctx.kernel_luma);
    clReleaseKernel(deshake->opencl_ctx.kernel_chroma);
    clReleaseProgram(deshake->opencl_ctx.program);
    deshake->opencl_ctx.command_queue = NULL;
    av_opencl_uninit();
}

int ff_opencl_deshake_process_inout_buf(AVFilterContext *ctx, AVFrame *in, AVFrame *out)
{
    int ret = 0;
    AVFilterLink *link = ctx->inputs[0];
    DeshakeContext *deshake = ctx->priv;
    const int hshift = av_pix_fmt_desc_get(link->format)->log2_chroma_h;
    int chroma_height = AV_CEIL_RSHIFT(link->h, hshift);

    if ((!deshake->opencl_ctx.cl_inbuf) || (!deshake->opencl_ctx.cl_outbuf)) {
        deshake->opencl_ctx.in_plane_size[0]  = (in->linesize[0] * in->height);
        deshake->opencl_ctx.in_plane_size[1]  = (in->linesize[1] * chroma_height);
        deshake->opencl_ctx.in_plane_size[2]  = (in->linesize[2] * chroma_height);
        deshake->opencl_ctx.out_plane_size[0] = (out->linesize[0] * out->height);
        deshake->opencl_ctx.out_plane_size[1] = (out->linesize[1] * chroma_height);
        deshake->opencl_ctx.out_plane_size[2] = (out->linesize[2] * chroma_height);
        deshake->opencl_ctx.cl_inbuf_size  = deshake->opencl_ctx.in_plane_size[0] +
                                             deshake->opencl_ctx.in_plane_size[1] +
                                             deshake->opencl_ctx.in_plane_size[2];
#ifdef OCL_FINDMOTION	
		deshake->opencl_ctx.cl_refbuf_size	= deshake->opencl_ctx.cl_inbuf_size;
#endif		
        deshake->opencl_ctx.cl_outbuf_size = deshake->opencl_ctx.out_plane_size[0] +
                                             deshake->opencl_ctx.out_plane_size[1] +
                                             deshake->opencl_ctx.out_plane_size[2];
        if (!deshake->opencl_ctx.cl_inbuf) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_inbuf,
                                            deshake->opencl_ctx.cl_inbuf_size,
											CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, NULL); //
            if (ret < 0)
                return ret;
        }
        if (!deshake->opencl_ctx.cl_outbuf) {
            ret = av_opencl_buffer_create(&deshake->opencl_ctx.cl_outbuf,
                                            deshake->opencl_ctx.cl_outbuf_size,
                                            CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, NULL); //
            if (ret < 0)
                return ret;
        }
    }
    ret = av_opencl_buffer_write_image(deshake->opencl_ctx.cl_inbuf,
                                 deshake->opencl_ctx.cl_inbuf_size,
                                 0, in->data,deshake->opencl_ctx.in_plane_size,
                                 deshake->opencl_ctx.plane_num);
    if(ret < 0)
        return ret;
    return ret;
}
