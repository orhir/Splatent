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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M, int M_f,
		const float* orig_points,
		const glm::vec3* scales,
		const float scale_modifier,
		const glm::vec4* rotations,
		const float* opacities,
		const float* shs,
		const float* feature_shs,
		bool* clamped,
		const float* cov3D_precomp,
		const float* colors_precomp,
		const float* features_precomp,
		const float* viewmatrix,
		const float* projmatrix,
		const float* projmatrix_inv,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float* features,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered,
		bool render_textured_features,
		int degree_feature);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H, int M_f,
		const float2* points_xy_image,
		const float* colors,
		const float* features,
		const float* render_noise,
		const float* depths,
		const float4* conic_opacity,
		const int* radii,
		float* out_alpha,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		float* out_feature_map,
		float* out_depth,
		bool use_slerp = false,
		float* out_render_noise = nullptr,
		float pixel_threshold = 1000.0f,
		const glm::vec3* means3D = nullptr,
		const float* cov3Ds = nullptr,
		const float* viewmatrix = nullptr,
		const float* projmatrix = nullptr,
		const float* projmatrix_inv = nullptr,
		const glm::vec3* cam_pos = nullptr,
		int degree_noise = 3,
		int blend_mode = 0,
		bool render_textured_features = false,
		int degree_feature = 3); 
}


#endif
