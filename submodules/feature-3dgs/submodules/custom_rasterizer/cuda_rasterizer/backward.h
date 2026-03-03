/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int M_f,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* features,
		const float* render_noise,
		const float* alphas,
		const float* depths,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dfeaturepixels,
		const float* dL_dalphas,
		const float* dL_dpixel_depths,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		float* dL_dfeatures,
		float* dL_ddepths,
		bool use_slerp = false,
		const float* dL_drender_noise = nullptr,
		float* dL_drender_noise_gaussians = nullptr,
		int blend_mode = 0,
		bool render_textured_features = false,
		int degree_feature = 3,
		const glm::vec3* means3D = nullptr,
		const float* cov3Ds = nullptr,
		const float* viewmatrix = nullptr,
		const float* projmatrix = nullptr,
		const float* projmatrix_inv = nullptr,
		const glm::vec3* cam_pos = nullptr,
		bool feature_backprop_geometry = false
	);

	void preprocess(
		int P, int D, int M, int M_f,
		const float3* means,
		const int* radii,
		const float* shs,
		const float* feature_shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dfeatures,
		float* dL_ddepth,
		float* dL_dcov3D,
		float* dL_dsh,
		float* dL_dfeature_shs,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		bool render_textured_features,
		int degree_feature,
		bool feature_backprop_geometry
	);
}

#endif
