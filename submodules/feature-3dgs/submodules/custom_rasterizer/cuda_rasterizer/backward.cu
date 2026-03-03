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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>
namespace cg = cooperative_groups;

// Helper functions for transformations (same as forward)
__forceinline__ __device__ float3 transformR_T(float3 v, const float R[9]) {
    return make_float3(
        R[0] * v.x + R[3] * v.y + R[6] * v.z,
        R[1] * v.x + R[4] * v.y + R[7] * v.z,
        R[2] * v.x + R[5] * v.y + R[8] * v.z
    );
}

__forceinline__ __device__ float4 transformFloat4_4x4(const float4& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12] * p.w,
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13] * p.w,
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14] * p.w,
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15] * p.w
	};
	return transformed;
}

__device__ static bool invert_3x3_symmetric(const float* cov, float* inv_cov) {
    float S00 = cov[0], S01 = cov[1], S02 = cov[2];
    float S11 = cov[3], S12 = cov[4], S22 = cov[5];
    
    float det = S00*(S11*S22 - S12*S12) - S01*(S01*S22 - S12*S02) + S02*(S01*S12 - S11*S02);
    
    if (fabsf(det) < 1e-10f) return false;
    
    float inv_det = 1.0f / det;
    inv_cov[0] = (S11*S22 - S12*S12) * inv_det;
    inv_cov[1] = (S02*S12 - S01*S22) * inv_det;
    inv_cov[2] = (S01*S12 - S02*S11) * inv_det;
    inv_cov[3] = inv_cov[1];
    inv_cov[4] = (S00*S22 - S02*S02) * inv_det;
    inv_cov[5] = (S01*S02 - S00*S12) * inv_det;
    inv_cov[6] = inv_cov[2];
    inv_cov[7] = inv_cov[5];
    inv_cov[8] = (S00*S11 - S01*S01) * inv_det;
    
    return true;
}



// This function is identical to forward pass - no changes needed
__device__ float3 getRayVec_b(float2 pix, int W, int H, const float* viewMatrix, const float* invProj, glm::vec3 campos)
{
	// Convert pixel coordinates to normalized device coordinates (NDC)
	float ndcX = 2.0f * ((pix.x + 1.0f) / W) - 1.0f;
	float ndcY = 2.0f * ((pix.y + 1.0f) / H) - 1.0f;

	// Define point in clip coordinates (z value chosen for a point in front of the camera)
	float p_hom_x_r = ndcX * 1.0000001f;
	float p_hom_y_r = ndcY * 1.0000001f;
	float p_hom_z_r = (100.0f + 0.01f - 1.0f) / (100.0f - 0.01f);
	float4 clipCoords = make_float4(p_hom_x_r, p_hom_y_r, p_hom_z_r, 1.0f);

	// Transform to camera space using the inverse projection matrix
	float4 camCoords = transformFloat4_4x4(clipCoords, invProj);
	float invW = 1.0f / camCoords.w;
	camCoords = make_float4(camCoords.x * invW, camCoords.y * invW, camCoords.z * invW, 1.0f);

	// Compute the direction vector from the camera position to the point in camera space
	float3 realVector = make_float3(camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z);

	// Normalize the direction vector
	float invNorm = 1.0f / sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
	float3 rayDirection = make_float3(realVector.x * invNorm, realVector.y * invNorm, realVector.z * invNorm);

	return rayDirection;
}


// Use the same intersection computation as forward pass
__device__ glm::vec3 getIntersection3D_b(float3 ray, const glm::vec3 mean, const float* covariance, glm::vec3 campos) {
    float inv_cov[9];
    if (!invert_3x3_symmetric(covariance, inv_cov)) {
        return glm::vec3(0.0f);
    }

    glm::vec3 v = campos - mean;
    float v_arr[3] = {v.x, v.y, v.z};
    float d_arr[3] = {ray.x, ray.y, ray.z};

    // Compute A, B, C more efficiently
    float A = d_arr[0]*(inv_cov[0]*d_arr[0] + inv_cov[1]*d_arr[1] + inv_cov[2]*d_arr[2]) +
              d_arr[1]*(inv_cov[3]*d_arr[0] + inv_cov[4]*d_arr[1] + inv_cov[5]*d_arr[2]) +
              d_arr[2]*(inv_cov[6]*d_arr[0] + inv_cov[7]*d_arr[1] + inv_cov[8]*d_arr[2]);

    float B = 2.0f * (v_arr[0]*(inv_cov[0]*d_arr[0] + inv_cov[1]*d_arr[1] + inv_cov[2]*d_arr[2]) +
                      v_arr[1]*(inv_cov[3]*d_arr[0] + inv_cov[4]*d_arr[1] + inv_cov[5]*d_arr[2]) +
                      v_arr[2]*(inv_cov[6]*d_arr[0] + inv_cov[7]*d_arr[1] + inv_cov[8]*d_arr[2]));

    float C = v_arr[0]*(inv_cov[0]*v_arr[0] + inv_cov[1]*v_arr[1] + inv_cov[2]*v_arr[2]) +
              v_arr[1]*(inv_cov[3]*v_arr[0] + inv_cov[4]*v_arr[1] + inv_cov[5]*v_arr[2]) +
              v_arr[2]*(inv_cov[6]*v_arr[0] + inv_cov[7]*v_arr[1] + inv_cov[8]*v_arr[2]) - 1.0f;

    float discriminant = B*B - 4.0f*A*C;
    if (discriminant < 0) return glm::vec3(0.0f);

    float t = (-B - sqrtf(discriminant)) / (2.0f * A);
    if (t <= 0) return glm::vec3(0.0f);

    glm::vec3 intersection_point = campos + t * glm::vec3(ray.x, ray.y, ray.z);
    glm::vec3 local_point = intersection_point - mean;
    
    // Normal = inv_cov * local_point
    glm::vec3 normal(
        inv_cov[0]*local_point.x + inv_cov[1]*local_point.y + inv_cov[2]*local_point.z,
        inv_cov[3]*local_point.x + inv_cov[4]*local_point.y + inv_cov[5]*local_point.z,
        inv_cov[6]*local_point.x + inv_cov[7]*local_point.y + inv_cov[8]*local_point.z
    );

    float len = glm::length(normal);
    return len > 1e-6f ? normal / len : glm::vec3(0.0f);
}


// Forward pass for feature computation - adapted from computeColorFromIntersection
__device__ void computeFeatureFromIntersection_f(int idx, const glm::vec3 unit_int, const float* texture, const int deg, const int max_coeffs, glm::vec4* final_result)
{
	const glm::vec4* sh = ((glm::vec4*)texture) + idx * max_coeffs;

	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;

	glm::vec4 result = HO_SH[0] * sh[0];

	if (deg > 0)
	{
			result += -HO_SH[9] * x * sh[1] +
						  HO_SH[9] * y * sh[2] +
						  -HO_SH[9] * z * sh[3];
	}

	// Add higher order terms as needed (same as forward pass)
	if (deg > 1)
	{
		float xx = x*x, yy = y*y, zz = z*z;
		float xz = x*z, xy = x*y, yz = y*z;
		float x_plus_z = x + z;
		result = result + HO_SH[28] * xz * sh[4] +
						-HO_SH[28] * xy * sh[5] +
						(-HO_SH[1]*xx + HO_SH[15]*yy - HO_SH[1]*zz) * sh[6] +
						-HO_SH[28] * yz * sh[7] +
						x_plus_z*(-HO_SH[12]*x + HO_SH[12]*z) * sh[8];

		if (deg > 2)
		{
			float xx = x*x, yy = y*y, zz = z*z;
			float xy = x*y, xz = x*z, yz = y*z;
			float x_minus_z = x - z, x_plus_z = x + z;
			float xxx = xx*x, yyy = yy*y, zzz = zz*z;
			
			result = result + 	(-HO_SH[32]*x*zz + HO_SH[13]*xxx) * sh[9] +
								HO_SH[45] * xz*y * sh[10] +
								(-HO_SH[33]*x*yy + HO_SH[6]*x*zz + HO_SH[6]*xxx) * sh[11] +
								-HO_SH[4] * y*(HO_SH[47]*xx - HO_SH[35]*yy + HO_SH[47]*zz) * sh[12] +
								(HO_SH[6]*xx*z - HO_SH[33]*yy*z + HO_SH[6]*zzz) * sh[13] +
								-HO_SH[29] * x_minus_z*x_plus_z*y * sh[14] +
								(HO_SH[32]*xx*z - HO_SH[13]*zzz) * sh[15];


			if (deg > 3)
			{

				float xxxx = xxx*x, yyyy = yyy*y, zzzz = zzz*z;
				float xxxy = xxx*y, xxxz = xxx*z, xyyyy = x*yyyy;
			
				result = result -HO_SH[40] * x_minus_z*x_plus_z*xz * sh[16] +
								(HO_SH[32]*xxxy - HO_SH[59]*xy*zz) * sh[17] +
								(-HO_SH[24]*x*zzz - HO_SH[24]*xxxz + HO_SH[62]*xz*yy) * sh[18] +
								HO_SH[18] * xy*(HO_SH[47]*xx - HO_SH[53]*yy + HO_SH[47]*zz) * sh[19] +
								(HO_SH[2]*xxxx - HO_SH[42]*xx*yy + HO_SH[16]*xx*zz + HO_SH[20]*yyyy - HO_SH[42]*yy*zz + HO_SH[2]*zzzz) * sh[20] +
								HO_SH[18] * yz*(HO_SH[47]*xx - HO_SH[53]*yy + HO_SH[47]*zz) * sh[21] +
								(HO_SH[8]*xxxx - HO_SH[44]*xx*yy + HO_SH[44]*yy*zz - HO_SH[8]*zzzz) * sh[22] +
								(HO_SH[59]*xx*yz - HO_SH[32]*y*zzz) * sh[23] +
								(HO_SH[14]*xxxx - HO_SH[51]*xx*zz + HO_SH[14]*zzzz) * sh[24];

				if (deg > 4)
				{
					float xxxxx = xxxx*x, yyyyy = yyyy*y, zzzzz = zzzz*z;
					float xxxzz = xxx*zz, xxxyy = xxx*yy;
					float xyyzz = x*yy*zz, xzzzz = x*zzzz, yzzzz = y*zzzz;
					float xyzzz = xy*zzz, xxyyy = xx*yyy, yyzzz = yy*zzz, xxzzz = xx*zzz;
					float xxyzz = xx*y*zz, xxxxz = xxxx*z, xxxyz = xxx * yz, xxxxy = xxxx * y, xxyyz = xx * yy * z;
					float yyyyz = yyyy * z, xzy = x*y*z, xyy = x * yy, xzyyy = xz * yyy, yyyzz = yyy * zz;

					result = 	result + 	(-HO_SH[48]*xzzzz + HO_SH[65]*xxxzz - HO_SH[17] * xxxxx) * sh[25] +
											(-HO_SH[69] * x_minus_z * x_plus_z * xzy) * sh[26] +
											(-HO_SH[73]*xyy*zz + HO_SH[30]*xzzzz + HO_SH[52]*xxxyy + HO_SH[26]*xxxzz - HO_SH[10]*xxxxx) * sh[27] +
											(-HO_SH[57]*xxxyz - HO_SH[57]*xyzzz + HO_SH[70]*xzyyy) * sh[28] +
											(-HO_SH[50]*xyyyy + HO_SH[60]*xyyzz - HO_SH[5]*xzzzz + HO_SH[60]*xxxyy - HO_SH[21]*xxxzz - HO_SH[5]*xxxxx) * sh[29] +
											(HO_SH[31]*xxxxy + HO_SH[49]*xxyzz - HO_SH[56]*xxyyy + HO_SH[31]*yzzzz - HO_SH[56]*yyyzz + HO_SH[23]*yyyyy) * sh[30] +
											(-HO_SH[5]*xxxxz + HO_SH[60]*xxyyz - HO_SH[21]*xxzzz - HO_SH[50]*yyyyz + HO_SH[60]*yyzzz - HO_SH[5]*zzzzz) * sh[31] +
											(HO_SH[39]*xxxxy - HO_SH[57]*xxyyy - HO_SH[39]*yzzzz + HO_SH[57]*yyyzz) * sh[32] +
											(-HO_SH[30]*xxxxz + HO_SH[73]*xxyyz - HO_SH[26]*xxzzz - HO_SH[52]*yyzzz + HO_SH[10]*zzzzz) * sh[33] +
											(HO_SH[37]*xxxxy - HO_SH[75]*xxyzz + HO_SH[37]*yzzzz) * sh[34] +
											(-HO_SH[48]*xxxxz + HO_SH[65]*xxzzz - HO_SH[17]*zzzzz) * sh[35];

					if (deg > 5)
					{
						// Add degree 6 terms - simplified version
						for (int coeff = 36; coeff < 49; coeff++) {
							result = result + 0.0f * sh[coeff];
					}
				}
			}
		}
	}
	}
	*final_result = result;
}


// Backward pass for feature computation
__device__ void computeFeatureFromIntersection_b(int idx, const glm::vec3 unit_int, const float* dL_dfeatures_in, float* dL_dfeatures_out, const int deg, const int max_coeffs)
{	
	float x = unit_int.x;
	float y = unit_int.y;
	float z = unit_int.z;
	float* dL_dfeature_out = dL_dfeatures_out + idx * max_coeffs * NUM_FEATURE_CHANNELS;

	// Compute gradients for each feature channel
	const glm::vec4 dL_dFeat = *((glm::vec4*)dL_dfeatures_in);
	glm::vec4 dL_dfeature[36];

	// Degree 0
	dL_dfeature[0] = HO_SH[0] * dL_dFeat;
		
	if (deg > 0) {
		dL_dfeature[1] = -HO_SH[9] * x * dL_dFeat;
		dL_dfeature[2] = HO_SH[9] * y * dL_dFeat;
		dL_dfeature[3] = -HO_SH[9] * z * dL_dFeat;

		if (deg > 1) {
			float xx = x*x, yy = y*y, zz = z*z;
			float xz = x*z, xy = x*y, yz = y*z;
			float x_plus_z = x + z;

			dL_dfeature[4] = HO_SH[28] * xz * dL_dFeat;
			dL_dfeature[5] = -HO_SH[28] * xy * dL_dFeat;
			dL_dfeature[6] = (-HO_SH[1]*xx + HO_SH[15]*yy - HO_SH[1]*zz) * dL_dFeat;
			dL_dfeature[7] = -HO_SH[28] * yz * dL_dFeat;
			dL_dfeature[8] = x_plus_z*(-HO_SH[12]*x + HO_SH[12]*z) * dL_dFeat;

			if (deg > 2) {	
				float x_minus_z = x - z;
				float xxx = xx*x, yyy = yy*y, zzz = zz*z;

				dL_dfeature[9] = (-HO_SH[32]*x*zz + HO_SH[13]*xxx) * dL_dFeat;
				dL_dfeature[10] = HO_SH[45] * xz*y * dL_dFeat;
				dL_dfeature[11] = (-HO_SH[33]*x*yy + HO_SH[6]*x*zz + HO_SH[6]*xxx) * dL_dFeat;
				dL_dfeature[12] = -HO_SH[4] * y *(HO_SH[47]*xx - HO_SH[35]*yy + HO_SH[47]*zz) * dL_dFeat;
				dL_dfeature[13] = (HO_SH[6]*xx*z - HO_SH[33]*yy*z + HO_SH[6]*zzz) * dL_dFeat;
				dL_dfeature[14] = -HO_SH[29] * x_minus_z*x_plus_z*y * dL_dFeat;
				dL_dfeature[15] = (HO_SH[32]*xx*z - HO_SH[13]*zzz) * dL_dFeat;

				if (deg > 3) {
					float xxxx = xxx*x, yyyy = yyy*y, zzzz = zzz*z;
					float xxxy = xxx*y, xxxz = xxx*z;
					
					dL_dfeature[16] = -HO_SH[40] * x_minus_z*x_plus_z*xz * dL_dFeat;
					dL_dfeature[17] = (HO_SH[32]*xxxy - HO_SH[59]*xy*zz) * dL_dFeat;
					dL_dfeature[18] = (-HO_SH[24]*x*zzz - HO_SH[24]*xxxz + HO_SH[62]*xz*yy) * dL_dFeat;
					dL_dfeature[19] = HO_SH[18] * xy*(HO_SH[47]*xx - HO_SH[53]*yy + HO_SH[47]*zz) * dL_dFeat;
					dL_dfeature[20] = (HO_SH[2]*xxxx - HO_SH[42]*xx*yy + HO_SH[16]*xx*zz + HO_SH[20]*yyyy - HO_SH[42]*yy*zz + HO_SH[2]*zzzz) * dL_dFeat;
					dL_dfeature[21] = HO_SH[18] * yz*(HO_SH[47]*xx - HO_SH[53]*yy + HO_SH[47]*zz) * dL_dFeat;
					dL_dfeature[22] = (HO_SH[8]*xxxx - HO_SH[44]*xx*yy + HO_SH[44]*yy*zz - HO_SH[8]*zzzz) * dL_dFeat;
					dL_dfeature[23] = (HO_SH[59]*xx*yz - HO_SH[32]*y*zzz) * dL_dFeat;
					dL_dfeature[24] = (HO_SH[14]*xxxx - HO_SH[51]*xx*zz + HO_SH[14]*zzzz) * dL_dFeat;

					if (deg > 4) {
						float xxxxx = xxxx*x, yyyyy = yyyy*y, zzzzz = zzzz*z;
						float xxxzz = xxx*zz, xxxyy = xxx*yy;
						float xyyzz = x*yy*zz, xzzzz = x*zzzz, yzzzz = y*zzzz;
						float xyzzz = xy*zzz, xxyyy = xx*yyy, yyzzz = yy*zzz, xxzzz = xx*zzz;
						float xyyyy = x*yyyy, yyyyz = yyyy * z;
						float xxyzz = xx*y*zz, xxxxz = xxxx*z, xxxyz = xxx * yz, xxxxy = xxxx * y, xxyyz = xx * yy * z;

						dL_dfeature[25] = (-HO_SH[48]*xzzzz + HO_SH[65]*xxxzz - HO_SH[17]*xxxxx) * dL_dFeat;
						dL_dfeature[26] = -HO_SH[69] * x_minus_z*x_plus_z*xz*y * dL_dFeat;
						dL_dfeature[27] = (-HO_SH[73]*xyyzz + HO_SH[30]*xzzzz + HO_SH[52]*xxxyy + HO_SH[26]*xxxzz - HO_SH[10]*xxxxx) * dL_dFeat;
						dL_dfeature[28] = (-HO_SH[57]*xxxyz - HO_SH[57]*xyzzz + HO_SH[70]*xz*yyy) * dL_dFeat;
						dL_dfeature[29] = (-HO_SH[50]*xyyyy + HO_SH[60]*xyyzz - HO_SH[5]*xzzzz + HO_SH[60]*xxxyy - HO_SH[21]*xxxzz - HO_SH[5]*xxxxx) * dL_dFeat;
						dL_dfeature[30] = (HO_SH[31]*xxxxy + HO_SH[49]*xxyzz - HO_SH[56]*xxyyy + HO_SH[31]*yzzzz - HO_SH[56]*yyzzz + HO_SH[23]*yyyyy) * dL_dFeat;
						dL_dfeature[31] = (-HO_SH[5]*xxxxz + HO_SH[60]*xxyyz - HO_SH[21]*xxzzz - HO_SH[50]*yyyyz + HO_SH[60]*yyzzz - HO_SH[5]*zzzzz) * dL_dFeat;
						dL_dfeature[32] = (HO_SH[39]*xxxxy - HO_SH[57]*xxyyy - HO_SH[39]*yzzzz + HO_SH[57]*yyzzz) * dL_dFeat;
						dL_dfeature[33] = (-HO_SH[30]*xxxxz + HO_SH[73]*xxyyz - HO_SH[26]*xxzzz - HO_SH[52]*yyzzz + HO_SH[10]*zzzzz) * dL_dFeat;
						dL_dfeature[34] = (HO_SH[37]*xxxxy - HO_SH[75]*xxyzz + HO_SH[37]*yzzzz) * dL_dFeat;
						dL_dfeature[35] = (-HO_SH[48]*xxxxz + HO_SH[65]*xxzzz - HO_SH[17]*zzzzz) * dL_dFeat;

					}
				}
			}
		}
	}
	for (int i = 0; i < (deg+1)*(deg+1); i++)
	{

		atomicAdd(&dL_dfeature_out[4*i], dL_dfeature[i].x);
		atomicAdd(&dL_dfeature_out[4*i + 1], dL_dfeature[i].y);
		atomicAdd(&dL_dfeature_out[4*i + 2], dL_dfeature[i].z);
		atomicAdd(&dL_dfeature_out[4*i + 3], dL_dfeature[i].w);

	}
}


// N-dimensional SLERP backward computation helper function for FEATURES only
__device__ void computeFeatureSlerpBackwardND(
    const float* prev_feat, const float* curr_feat, float alpha, 
    const float* grad_output, float* grad_prev, float* grad_curr, int dim) {
    
    // Calculate norms
    float norm_prev = 0.0f, norm_curr = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_prev += prev_feat[i] * prev_feat[i];
        norm_curr += curr_feat[i] * curr_feat[i];
    }
    norm_prev = sqrtf(norm_prev);
    norm_curr = sqrtf(norm_curr);
    
    // Handle degenerate cases - fallback to linear interpolation
    if (norm_prev < 1e-6f || norm_curr < 1e-6f) {
        for (int i = 0; i < dim; i++) {
            grad_prev[i] = (1.0f - alpha) * grad_output[i];
            grad_curr[i] = alpha * grad_output[i];
        }
        return;
    }
    
    // Normalize vectors
    float n_prev[4], n_curr[4];  // Support up to 4D
    for (int i = 0; i < dim; i++) {
        n_prev[i] = prev_feat[i] / norm_prev;
        n_curr[i] = curr_feat[i] / norm_curr;
    }
    
    // Compute dot product of normalized vectors
    float dot_product = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot_product += n_prev[i] * n_curr[i];
    }
    dot_product = fminf(1.0f, fmaxf(-1.0f, dot_product)); // Clamp to avoid numerical issues
    
    // Handle nearly parallel vectors - fallback to linear interpolation
    if (fabsf(dot_product) > 0.9995f) {
        for (int i = 0; i < dim; i++) {
            grad_prev[i] = (1.0f - alpha) * grad_output[i];
            grad_curr[i] = alpha * grad_output[i];
        }
        return;
    }
    
    // Compute SLERP angle
    float theta = acosf(fabsf(dot_product));
    float sin_theta = sinf(theta);
    
    if (sin_theta < 1e-6f) {
        // Fallback to linear interpolation gradients
        for (int i = 0; i < dim; i++) {
            grad_prev[i] = (1.0f - alpha) * grad_output[i];
            grad_curr[i] = alpha * grad_output[i];
        }
        return;
    }
    
    // SLERP coefficients (handle negative dot product by flipping curr)
    float sign = (dot_product >= 0.0f) ? 1.0f : -1.0f;
    float coeff_prev = sinf((1.0f - alpha) * theta) / sin_theta;
    float coeff_curr = sign * sinf(alpha * theta) / sin_theta;
    
    // Compute derivatives of coefficients w.r.t. theta
    float dcoeff_prev_dtheta = ((1.0f - alpha) * cosf((1.0f - alpha) * theta) - 
                               coeff_prev * cosf(theta)) / sin_theta;
    float dcoeff_curr_dtheta = sign * (alpha * cosf(alpha * theta) - 
                               coeff_curr * cosf(theta) / sign) / sin_theta;
    
    // Compute dtheta/d(dot_product): d(arccos(|x|))/dx = -sign(x)/√(1-x²)
    float dtheta_ddot = -(dot_product >= 0.0f ? 1.0f : -1.0f) / sqrtf(1.0f - dot_product * dot_product);
    
    // Compute scalar term for indirect gradients
    float scalar_term = 0.0f;
    for (int j = 0; j < dim; j++) {
        scalar_term += grad_output[j] * (dcoeff_prev_dtheta * n_prev[j] + dcoeff_curr_dtheta * n_curr[j]);
    }
    float indirect_multiplier = scalar_term * dtheta_ddot;
    
    // Full SLERP gradients including indirect terms
    for (int i = 0; i < dim; i++) {
        // Direct gradient terms
        float grad_prev_direct = coeff_prev * grad_output[i];
        float grad_curr_direct = coeff_curr * grad_output[i];
        
        // Derivatives of dot product w.r.t. input features
        float ddot_dprev = (n_curr[i] / norm_prev) - (dot_product * prev_feat[i]) / (norm_prev * norm_prev);
        float ddot_dcurr = (n_prev[i] / norm_curr) - (dot_product * curr_feat[i]) / (norm_curr * norm_curr);
        
        // Indirect gradient terms through coefficient derivatives
        float grad_prev_indirect = indirect_multiplier * ddot_dprev;
        float grad_curr_indirect = indirect_multiplier * ddot_dcurr;
        
        // Combine direct and indirect terms
        grad_prev[i] = grad_prev_direct + grad_prev_indirect;
        grad_curr[i] = grad_curr_direct + grad_curr_indirect;
    }
}

// Backward pass for conversion of spherical harmonics to features for
// each Gaussian.
__device__ void computeFeatureFromSH(
	int idx, 
	int deg, 
	int max_coeffs, 
	const glm::vec3* means, 
	glm::vec3 campos, 
	const float* shs, 
	const glm::vec4* dL_dfeatures, 
	glm::vec3* dL_dmeans, 
	glm::vec4* dL_dshs,
	bool feature_backprop_geometry)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	// Feature SH coefficients are stored as [gaussian_idx][coeff_idx][feature_ch]
	glm::vec4 dL_dfeature = dL_dfeatures[idx];
	glm::vec4* sh = ((glm::vec4*)shs) + idx * max_coeffs;
	glm::vec4* dL_dsh = dL_dshs + idx * max_coeffs;

	// No clamping for features (unlike RGB)
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Gradients w.r.t. direction for geometry backprop
	glm::vec4 dFeatdx(0, 0, 0, 0);
	glm::vec4 dFeatdy(0, 0, 0, 0);
	glm::vec4 dFeatdz(0, 0, 0, 0);

	// Degree 0
	dL_dsh[0] = SH_C0 * dL_dfeature;

	if (deg > 0) {
			// Degree 1
			dL_dsh[1] = -SH_C1 * x * dL_dfeature;
			dL_dsh[2] = SH_C1 * y * dL_dfeature;
			dL_dsh[3] = -SH_C1 * z * dL_dfeature;

			if (feature_backprop_geometry) {
				dFeatdx = -SH_C1 * sh[1];
				dFeatdy = SH_C1 * sh[2];
				dFeatdz = -SH_C1 * sh[3];
			}

			if (deg > 1) {
				// Degree 2
				float zz = z * z, xx = x * x, yy = y * y;
				float zx = z * x, xy = x * y, zy = z * y;

				dL_dsh[4] = SH_C2[0] * zx * dL_dfeature;
				dL_dsh[5] = SH_C2[1] * xy * dL_dfeature;
				dL_dsh[6] = SH_C2[2] * (2.0f * yy - zz - xx) * dL_dfeature;
				dL_dsh[7] = SH_C2[3] * zy * dL_dfeature;
				dL_dsh[8] = SH_C2[4] * (zz - xx) * dL_dfeature;

				if (feature_backprop_geometry) {
					dFeatdx += SH_C2[0] * z * sh[4] +
								   SH_C2[1] * y * sh[5] +
								   SH_C2[2] * 2.0f * -x * sh[6] +
								   SH_C2[4] * 2.0f * -x * sh[8];
					dFeatdy += SH_C2[1] * x * sh[5] +
								   SH_C2[2] * 2.0f * 2.0f * y * sh[6] +
								   SH_C2[3] * z * sh[7];
					dFeatdz += SH_C2[0] * x * sh[4] +
								   SH_C2[2] * 2.0f * -z * sh[6] +
								   SH_C2[3] * y * sh[7] +
								   SH_C2[4] * 2.0f * z * sh[8];
				}

				if (deg > 2) {
					// Degree 3
					dL_dsh[9] += SH_C3[0] * x * (3.0f * zz - xx) * dL_dfeature;
					dL_dsh[10] += SH_C3[1] * zx * y * dL_dfeature;
					dL_dsh[11] += SH_C3[2] * x * (4.0f * yy - zz - xx) * dL_dfeature;
					dL_dsh[12] += SH_C3[3] * y * (2.0f * yy - 3.0f * zz - 3.0f * xx) * dL_dfeature;
					dL_dsh[13] += SH_C3[4] * z * (4.0f * yy - zz - xx) * dL_dfeature;
					dL_dsh[14] += SH_C3[5] * z * (zz - xx) * dL_dfeature;
					dL_dsh[15] += SH_C3[6] * z * (zz - 3.0f * xx) * dL_dfeature;

					if (feature_backprop_geometry) {
						dFeatdx += SH_C3[0] * sh[9] * 3.0f * (zz - xx) +
									   SH_C3[1] * sh[10] * zy +
									   SH_C3[2] * sh[11] * (-3.0f * xx + 4.0f * yy - zz) +
									   SH_C3[3] * sh[12] * -3.0f * 2.0f * xy +
									   SH_C3[4] * sh[13] * -2.0f * zx +
									   SH_C3[5] * sh[14] * -2.0f * xy +
									   SH_C3[6] * sh[15] * -3.0f * 2.0f * zx;
						dFeatdy += SH_C3[1] * sh[10] * zx +
									   SH_C3[2] * sh[11] * 4.0f * 2.0f * xy +
									   SH_C3[3] * sh[12] * 3.0f * (2.0f * yy - zz - xx) +
									   SH_C3[4] * sh[13] * 4.0f * 2.0f * zy +
									   SH_C3[5] * sh[14] * (zz - xx);
						dFeatdz += SH_C3[0] * sh[9] * 3.0f * 2.0f * zx +
									   SH_C3[1] * sh[10] * xy +
									   SH_C3[2] * sh[11] * -2.0f * zx +
									   SH_C3[3] * sh[12] * -3.0f * 2.0f * zy +
									   SH_C3[4] * sh[13] * (-3.0f * zz + 4.0f * yy - xx) +
									   SH_C3[5] * sh[14] * 2.0f * zy +
									   SH_C3[6] * sh[15] * 3.0f * (zz - xx);
				}
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so feature gradients
	// must propagate back into 3D position (only if feature_backprop_geometry is true).
	if (feature_backprop_geometry) {
		glm::vec3 dL_ddir(glm::dot(dFeatdx, dL_dfeature), glm::dot(dFeatdy, dL_dfeature), glm::dot(dFeatdz, dL_dfeature));
		// Account for normalization of direction
		float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

		// Gradients of loss w.r.t. Gaussian means, but only the portion 
		// that is caused because the mean affects the view-dependent features.
		dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
		
	}
}

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(
	int idx, 
	int deg, 
	int max_coeffs, 
	const glm::vec3* means, 
	glm::vec3 campos, 
	const float* shs, 
	const bool* clamped, 
	const glm::vec3* dL_dcolor, 
	glm::vec3* dL_dmeans, 
	glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdz(0, 0, 0);
	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * x;
		float dRGBdsh2 = SH_C1 * y;
		float dRGBdsh3 = -SH_C1 * z;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[1];
		dRGBdy = SH_C1 * sh[2];
		dRGBdz = -SH_C1 * sh[3];

		if (deg > 1)
		{
			float zz = z * z, xx = x * x, yy = y * y;
			float zx = z * x, xy = x * y, zy = z * y;

			float dRGBdsh4 = SH_C2[0] * zx;
			float dRGBdsh5 = SH_C2[1] * xy;
			float dRGBdsh6 = SH_C2[2] * (2.f * yy - zz - xx);
			float dRGBdsh7 = SH_C2[3] * zy;
			float dRGBdsh8 = SH_C2[4] * (zz - xx);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdz += SH_C2[0] * x * sh[4] + SH_C2[2] * 2.f * -z * sh[6] + SH_C2[3] * y * sh[7] + SH_C2[4] * 2.f * z * sh[8];
			dRGBdx += SH_C2[0] * z * sh[4] + SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[4] * 2.f * -x * sh[8];
			dRGBdy += SH_C2[1] * x * sh[5] + SH_C2[2] * 2.f * 2.f * y * sh[6] + SH_C2[3] * z * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * x * (3.f * zz - xx);
				float dRGBdsh10 = SH_C3[1] * zx * y;
				float dRGBdsh11 = SH_C3[2] * x * (4.f * yy - zz - xx);
				float dRGBdsh12 = SH_C3[3] * y * (2.f * yy - 3.f * zz - 3.f * xx);
				float dRGBdsh13 = SH_C3[4] * z * (4.f * yy - zz - xx);
				float dRGBdsh14 = SH_C3[5] * y * (zz - xx);
				float dRGBdsh15 = SH_C3[6] * z * (zz - 3.f * xx);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdz += (
					SH_C3[0] * sh[9] * 3.f * 2.f * zx +
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * -2.f * zx +
					SH_C3[3] * sh[12] * -3.f * 2.f * zy +
					SH_C3[4] * sh[13] * (-3.f * zz + 4.f * yy - xx) +
					SH_C3[5] * sh[14] * 2.f * zy +
					SH_C3[6] * sh[15] * 3.f * (zz - xx));

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * (zz - xx) +
					SH_C3[1] * sh[10] * zy +
					SH_C3[2] * sh[11] * (-3.f * xx + 4.f * yy - zz) +
					SH_C3[3] * sh[12] * -3.f * 2.f * xy +
					SH_C3[4] * sh[13] * -2.f * zx +
					SH_C3[5] * sh[14] * -2.f * xy +
					SH_C3[6] * sh[15] * -3.f * 2.f * zx);

				dRGBdy += (
					SH_C3[1] * sh[10] * zx +
					SH_C3[2] * sh[11] * 4.f * 2.f * xy +
					SH_C3[3] * sh[12] * 3.f * (2.f * yy - zz - xx) +
					SH_C3[4] * sh[13] * 4.f * 2.f * zy +
					SH_C3[5] * sh[14] * (zz - xx));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M, int M_f,
	const float3* means,
	const int* radii,
	const float* shs,
	const float* feature_shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* view,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
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
	bool feature_backprop_geometry)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// the w must be equal to 1 for view^T * [x,y,z,1]
	float3 m_view = transformPoint4x3(m, view);

	// Compute loss gradient w.r.t. 3D means due to gradients of depth
	// from rendering procedure
	glm::vec3 dL_dmean2;
	float mul3 = view[2] * m.x + view[6] * m.y + view[10] * m.z + view[14];
	dL_dmean2.x = (view[2] - view[3] * mul3) * dL_ddepth[idx];
	dL_dmean2.y = (view[6] - view[7] * mul3) * dL_ddepth[idx];
	dL_dmean2.z = (view[10] - view[11] * mul3) * dL_ddepth[idx];

	// That's the third part of the mean gradient.
	dL_dmeans[idx] += dL_dmean2;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);
	
	// Compute gradient updates due to computing features from SHs (when textured_features is false)
	if (feature_shs && !render_textured_features)
		computeFeatureFromSH(idx, degree_feature, M_f, (glm::vec3*)means, *campos, feature_shs, (glm::vec4*)dL_dfeatures, (glm::vec3*)dL_dmeans, (glm::vec4*)dL_dfeature_shs, feature_backprop_geometry);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// Backward version of the rendering procedure.
// Assumes that colors is not null, because of only gradient to alpha
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int M_f,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ features,
	const float* __restrict__ render_noise,
	const float* __restrict__ alphas,
	const float* __restrict__ depths,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_dfeaturepixels,
	const float* __restrict__ dL_dalphas,
	const float* __restrict__ dL_dpixel_depths,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dfeatures,
	float* __restrict__ dL_ddepths,
	bool use_slerp,
	const float* __restrict__ dL_drender_noise,
	float* __restrict__ dL_drender_noise_gaussians,
	int blend_mode,
	bool render_textured_features,
	int degree_feature,
	const glm::vec3* __restrict__ means3D,
	const float* __restrict__ cov3Ds,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	const float* __restrict__ projmatrix_inv,
	const glm::vec3* __restrict__ cam_pos,
	bool feature_backprop_geometry) 
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float collected_depths[BLOCK_SIZE];
	__shared__ float collected_features[NUM_FEATURE_CHANNELS * BLOCK_SIZE];
	__shared__ float collected_cov3D[BLOCK_SIZE * 6];
	__shared__ glm::vec3 collected_mean[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? (1 - alphas[pix_id]) : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;
	
	float accum_rec[C] = { 0 };
	float accum_feat_rec[NUM_FEATURE_CHANNELS] = { 0 };
	float dL_dpixel[C];
	float dL_dfeaturepixel[NUM_FEATURE_CHANNELS];
	float accum_alpha_rec = 0;
	float dL_dalpha;
	float accum_depth_rec = 0;
	float dL_dpixel_depth;
	if (inside) {
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

		dL_dalpha = dL_dalphas[pix_id];
		dL_dpixel_depth = dL_dpixel_depths[pix_id];

		if (features)
			for (int i = 0; i < NUM_FEATURE_CHANNELS; i++) 
				dL_dfeaturepixel[i] = dL_dfeaturepixels[i * H * W + pix_id];
	}

	// Note: render_noise gradients are variance-preserving noise and don't propagate to other variables
	// (removed unused dL_dnoise variable)

	float last_alpha = 0;
	float last_color[C] = { 0 };
	float last_depth = 0;
	float last_feature[NUM_FEATURE_CHANNELS] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	float3 ray;
	if (render_textured_features)
		ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
			for (int i = 0; i < NUM_FEATURE_CHANNELS; i++)
				collected_features[i * BLOCK_SIZE + block.thread_rank()] = features[coll_id * NUM_FEATURE_CHANNELS + i];
			collected_depths[block.thread_rank()] = depths[coll_id];
			collected_mean[block.thread_rank()] = means3D[coll_id];
			for (int i = 0; i < 6; i++) {
				collected_cov3D[i * BLOCK_SIZE + block.thread_rank()] = cov3Ds[coll_id * 6 + i];
			}
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dopa = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dopa += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			if (features) {
				if (blend_mode == GAUSSIAN_SPLATTING) {
					// Check if we should use textured features backward
					if (render_textured_features) {
						// Textured feature backward pass
						glm::vec3 mean = collected_mean[j];
						const float* cov3D = &collected_cov3D[j];
						glm::vec3 unit_int = getIntersection3D_b(ray, mean, cov3D, *cam_pos);
						
						int degree = degree_feature;
						if (glm::length(unit_int) < 1e-6f)
							degree = 0;

						float tmp[NUM_FEATURE_CHANNELS];
						for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
							tmp[ch] = dchannel_dcolor * dL_dfeaturepixel[ch];
						}    
						computeFeatureFromIntersection_b(global_id, unit_int, tmp, dL_dfeatures, degree, M_f);
						
						// Add dL_dopa calculation for geometry backprop
						if (feature_backprop_geometry) {
							glm::vec4 textured_features = {0, 0, 0, 0};
							computeFeatureFromIntersection_f(global_id, unit_int, features, degree, M_f, &textured_features);
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								float f = 0;
								if (ch == 0)
									f = textured_features.x;
								else if (ch == 1)
									f = textured_features.y;
								else if (ch == 2)
									f = textured_features.z;
								else
									f = textured_features.w;
								accum_feat_rec[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_feat_rec[ch];
								last_feature[ch] = f;
								dL_dopa += (f - accum_feat_rec[ch]) * dL_dfeaturepixel[ch];
							}
						}
					} else {
						// Standard Gaussian splatting backward for features
						if (use_slerp) {
							float accum_feat[NUM_FEATURE_CHANNELS];
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								accum_feat[ch] = 0.0f;
							}
							float grad_out_feat[NUM_FEATURE_CHANNELS];
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								grad_out_feat[ch] = dchannel_dcolor * dL_dfeaturepixel[ch];
							}
							float grad_accum[NUM_FEATURE_CHANNELS], grad_curr[NUM_FEATURE_CHANNELS];
							computeFeatureSlerpBackwardND(accum_feat, &collected_features[0 * BLOCK_SIZE + j], alpha, grad_out_feat, grad_accum, grad_curr, NUM_FEATURE_CHANNELS);
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								atomicAdd(&(dL_dfeatures[global_id * NUM_FEATURE_CHANNELS + ch]), grad_curr[ch]);
							}
						} else {
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								const float f = collected_features[ch * BLOCK_SIZE + j];
								// Update last feature (to be used in the next iteration)
								accum_feat_rec[ch] = last_alpha * last_feature[ch] + (1.f - last_alpha) * accum_feat_rec[ch];
								last_feature[ch] = f;
								
								const float dL_dfeature_channel = dL_dfeaturepixel[ch];						
								// Feature gradient to opacity - corrected to match RGB pattern
								if (feature_backprop_geometry) {
									dL_dopa += (f - accum_feat_rec[ch]) * dL_dfeature_channel;
								}
        
								atomicAdd(&(dL_dfeatures[global_id * NUM_FEATURE_CHANNELS + ch]), dchannel_dcolor * dL_dfeature_channel);
							}
						}
					}
				} else {
					// Extended blending modes backward - recompute forward to get selection
					struct Contribution {
						glm::vec4 feature;
						float effective_alpha;
						int id;
					};
					Contribution contribs[256];
					int num_contribs = 0;
					
					// Extended blending modes use different feature computation - must match forward pass exactly
					for (int k = 0; k < min(BLOCK_SIZE, toDo) && num_contribs < 256; k++) {
						int k_id = collected_id[k];
						float2 k_xy = collected_xy[k];
						float2 k_d = { k_xy.x - pixf.x, k_xy.y - pixf.y };
						float4 k_con_o = collected_conic_opacity[k];
						float k_power = -0.5f * (k_con_o.x * k_d.x * k_d.x + k_con_o.z * k_d.y * k_d.y) - k_con_o.y * k_d.x * k_d.y;
						if (k_power > 0.0f) continue;
						float k_alpha = min(0.99f, k_con_o.w * exp(k_power));
						if (k_alpha < 1.0f / 255.0f) continue;
						
						contribs[num_contribs].effective_alpha = k_alpha;
						contribs[num_contribs].id = k_id;
						
						// Get features - must match forward pass computation
						if (render_textured_features) {
							float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);
							glm::vec3 mean = means3D[k_id];
							const float* cov3D = &cov3Ds[k_id * 6];
							glm::vec3 unit_int = getIntersection3D_b(ray, mean, cov3D, *cam_pos);
							
							if (glm::length(unit_int) > 1e-6f) {
								computeFeatureFromIntersection_f(k_id, unit_int, features, degree_feature, M_f, &contribs[num_contribs].feature);
							} else {
								for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
									contribs[num_contribs].feature[ch] = 0.0f;
								}
							}
						} else {
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								contribs[num_contribs].feature[ch] = collected_features[ch * BLOCK_SIZE + k];
							}
						}
						num_contribs++;
					}
					
					if (num_contribs > 0) {
						switch (blend_mode) {
							case ARGMAX_EFFECTIVE: {
								int max_idx = 0;
								for (int i = 1; i < num_contribs; i++) {
									if (contribs[i].effective_alpha > contribs[max_idx].effective_alpha) {
										max_idx = i;
									}
								}
								if (render_textured_features) {
									float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);
									glm::vec3 mean = means3D[contribs[max_idx].id];
									const float* cov3D = &cov3Ds[contribs[max_idx].id * 6];
									glm::vec3 unit_int = getIntersection3D_b(ray, mean, cov3D, *cam_pos);
									if (glm::length(unit_int) > 1e-6f) {
										computeFeatureFromIntersection_b(contribs[max_idx].id, unit_int, dL_dfeaturepixel, dL_dfeatures, degree_feature, M_f);
									}
								} else {
									for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
										atomicAdd(&(dL_dfeatures[contribs[max_idx].id * NUM_FEATURE_CHANNELS + ch]), dL_dfeaturepixel[ch]);
									}
								}
								// Geometry backprop for ARGMAX mode
								if (feature_backprop_geometry && contribs[max_idx].id == global_id) {
									for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
										dL_dopa += dL_dfeaturepixel[ch] * contribs[max_idx].feature[ch];
									}
								}
								break;
							}
							case MEDIAN_EFFECTIVE: {
								for (int i = 0; i < num_contribs - 1; i++) {
									for (int k = 0; k < num_contribs - i - 1; k++) {
										if (contribs[k].effective_alpha < contribs[k + 1].effective_alpha) {
											Contribution temp = contribs[k];
											contribs[k] = contribs[k + 1];
											contribs[k + 1] = temp;
										}
									}
								}
								int median_idx = num_contribs / 2;
								if (render_textured_features) {
									float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);
									glm::vec3 mean = means3D[contribs[median_idx].id];
									const float* cov3D = &cov3Ds[contribs[median_idx].id * 6];
									glm::vec3 unit_int = getIntersection3D_b(ray, mean, cov3D, *cam_pos);
									if (glm::length(unit_int) > 1e-6f) {
										computeFeatureFromIntersection_b(contribs[median_idx].id, unit_int, dL_dfeaturepixel, dL_dfeatures, degree_feature, M_f);
									}
								} else {
									for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
										atomicAdd(&(dL_dfeatures[contribs[median_idx].id * NUM_FEATURE_CHANNELS + ch]), dL_dfeaturepixel[ch]);
									}
								}
								// Geometry backprop for MEDIAN mode
								if (feature_backprop_geometry && contribs[median_idx].id == global_id) {
									for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
										dL_dopa += dL_dfeaturepixel[ch] * contribs[median_idx].feature[ch];
									}
								}
								break;
							}
							case RMS_EFFECTIVE: {
								float total_weight = 0.0f;
								float final_feature[NUM_FEATURE_CHANNELS] = {0};
								for (int i = 0; i < num_contribs; i++) {
									total_weight += contribs[i].effective_alpha;
								}
								for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
									float weighted_sum = 0.0f;
									for (int i = 0; i < num_contribs; i++) {
										weighted_sum += contribs[i].feature[ch] * contribs[i].feature[ch] * contribs[i].effective_alpha;
									}
									final_feature[ch] = sqrtf(weighted_sum / total_weight);
								}
								for (int i = 0; i < num_contribs; i++) {
									// Geometry backprop for RMS: ∂L/∂alpha_i = Σ_ch dL_dfeaturepixel[ch] * (features[i][ch]² - final_feature[ch]²) / (final_feature[ch] * total_weight)
									if (feature_backprop_geometry && contribs[i].id == global_id) {
										for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
											if (final_feature[ch] > 1e-8f && total_weight > 1e-8f) {
												float grad_alpha = dL_dfeaturepixel[ch] * (contribs[i].feature[ch] * contribs[i].feature[ch] - final_feature[ch] * final_feature[ch]) / (final_feature[ch] * total_weight);
												dL_dopa += grad_alpha;
											}
										}
									}
								}
								if (render_textured_features) {
									for (int i = 0; i < num_contribs; i++) {
										float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);
										glm::vec3 mean = means3D[contribs[i].id];
										const float* cov3D = &cov3Ds[contribs[i].id * 6];
										glm::vec3 unit_int = getIntersection3D_b(ray, mean, cov3D, *cam_pos);
										if (glm::length(unit_int) > 1e-6f) {
											float grad_weight[NUM_FEATURE_CHANNELS];
											for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
												grad_weight[ch] = dL_dfeaturepixel[ch] * contribs[i].feature[ch] * contribs[i].effective_alpha / (final_feature[ch] * total_weight);
											}
											computeFeatureFromIntersection_b(contribs[i].id, unit_int, grad_weight, dL_dfeatures, degree_feature, M_f);
										}
									}
								} else {
									for (int i = 0; i < num_contribs; i++) {
										for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
											float grad_feature = dL_dfeaturepixel[ch] * contribs[i].feature[ch] * contribs[i].effective_alpha / (final_feature[ch] * total_weight);
											atomicAdd(&(dL_dfeatures[contribs[i].id * NUM_FEATURE_CHANNELS + ch]), grad_feature);
										}
									}
								}
								break;
							}
							case SOFTMAX_EFFECTIVE: {
								// Recompute softmax weights (must match forward pass exactly)
								float max_alpha = contribs[0].effective_alpha;
								for (int i = 1; i < num_contribs; i++) {
									max_alpha = fmaxf(max_alpha, contribs[i].effective_alpha);
								}
								
								float exp_sum = 0.0f;
								float softmax_weights[256];
								float final_feature[NUM_FEATURE_CHANNELS] = {0};
								for (int i = 0; i < num_contribs; i++) {
									softmax_weights[i] = expf(contribs[i].effective_alpha - max_alpha);
									exp_sum += softmax_weights[i];
								}
								
								if (exp_sum > 1e-8f) {
									for (int i = 0; i < num_contribs; i++) {
										softmax_weights[i] /= exp_sum;
									}
									
									// Compute final feature for geometry backprop
									for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
										for (int i = 0; i < num_contribs; i++) {
											final_feature[ch] += softmax_weights[i] * contribs[i].feature[ch];
										}
									}
									
									for (int i = 0; i < num_contribs; i++) {
										// Geometry backprop for SOFTMAX: ∂L/∂alpha_i = Σ_ch dL_dfeaturepixel[ch] * softmax_weights[i] * (features[i][ch] - final_feature[ch])
										if (feature_backprop_geometry && contribs[i].id == global_id) {
											for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
												float grad_alpha_geom = dL_dfeaturepixel[ch] * softmax_weights[i] * (contribs[i].feature[ch] - final_feature[ch]);
												dL_dopa += grad_alpha_geom;
											}
										}
										
										// Propagate gradient to features (direct term)
										if (render_textured_features) {
											float3 ray = getRayVec_b(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);
											glm::vec3 mean = means3D[contribs[i].id];
											const float* cov3D = &cov3Ds[contribs[i].id * 6];
											glm::vec3 unit_int = getIntersection3D_b(ray, mean, cov3D, *cam_pos);
											if (glm::length(unit_int) > 1e-6f) {
												float grad_feature[NUM_FEATURE_CHANNELS];
												for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
													grad_feature[ch] = dL_dfeaturepixel[ch] * softmax_weights[i];
												}
												computeFeatureFromIntersection_b(contribs[i].id, unit_int, grad_feature, dL_dfeatures, degree_feature, M_f);
											}
										} else {
											for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
												atomicAdd(&(dL_dfeatures[contribs[i].id * NUM_FEATURE_CHANNELS + ch]), dL_dfeaturepixel[ch] * softmax_weights[i]);
											}
										}
									}
								}
								break;
							}
						}
					}
				}
			}

			
			// Propagate gradients from pixel depth to opacity
			const float c_d = collected_depths[j];
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dopa += (c_d - accum_depth_rec) * dL_dpixel_depth;
			atomicAdd(&(dL_ddepths[global_id]), dchannel_dcolor * dL_dpixel_depth);

			// Propagate gradients from pixel alpha (weights_sum) to opacity
			accum_alpha_rec = last_alpha + (1.f - last_alpha) * accum_alpha_rec;
			dL_dopa += (1 - accum_alpha_rec) * dL_dalpha;

			dL_dopa *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dopa += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			
			// For compatability to feature background
			float feature_bg_dot_dpixel = 0;
			for (int i = 0; i < NUM_FEATURE_CHANNELS; i++)
				feature_bg_dot_dpixel += 0.0f * dL_dfeaturepixel[i];
			dL_dopa += (-T_final / (1.f - alpha)) * feature_bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dopa;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dopa);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M, int M_f,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const float* feature_shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
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
	bool feature_backprop_geometry)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_COLOR_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M, M_f,
		(float3*)means3D,
		radii,
		shs,
		feature_shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dfeatures,
		dL_ddepth,
		dL_dcov3D,
		dL_dsh,
		dL_dfeature_shs,
		dL_dscale,
		dL_drot,
		render_textured_features,
		degree_feature,
		feature_backprop_geometry);
}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
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
	bool use_slerp,
	const float* dL_drender_noise,
	float* dL_drender_noise_gaussians,
	int blend_mode,
	bool render_textured_features,
	int degree_feature,
	const glm::vec3* means3D,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	bool feature_backprop_geometry) 
	
{
	renderCUDA<NUM_COLOR_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H, M_f,
		bg_color,
		means2D,
		conic_opacity,
		colors,
		features,
		render_noise,
		alphas,
		depths,
		n_contrib,
		dL_dpixels,
		dL_dfeaturepixels,
		dL_dalphas,
		dL_dpixel_depths,
		dL_dmean2D,
		dL_dconic2D,
		dL_dopacity,
		dL_dcolors,
		dL_dfeatures,
		dL_ddepths,
		use_slerp,
		dL_drender_noise,
		dL_drender_noise_gaussians,
		blend_mode,
		render_textured_features,
		degree_feature,
		means3D,
		cov3Ds,
		viewmatrix,
		projmatrix,
		projmatrix_inv,
		cam_pos,
		feature_backprop_geometry
		);
}
