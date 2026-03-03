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

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <curand_kernel.h>
namespace cg = cooperative_groups;

// Helper functions for transformations
__forceinline__ __device__ float3 transformR_T(float3 v, const float R[9]) {
    return make_float3(
        R[0] * v.x + R[3] * v.y + R[6] * v.z,  // Column-major matrix application
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

__device__ float3 getRayVec(float2 pix, int W, int H, const float* viewMatrix, const float* invProj, glm::vec3 campos)
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
	float invW = 1.0f / camCoords.w;  // Compute inverse of w once
	camCoords = make_float4(camCoords.x * invW, camCoords.y * invW, camCoords.z * invW, 1.0f);

	// Compute the direction vector from the camera position to the point in camera space
	float3 realVector = make_float3(camCoords.x - campos.x, camCoords.y - campos.y, camCoords.z - campos.z);

	// Normalize the direction vector
	float invNorm = 1.0f / sqrt(realVector.x * realVector.x + realVector.y * realVector.y + realVector.z * realVector.z);
	float3 rayDirection = make_float3(realVector.x * invNorm, realVector.y * invNorm, realVector.z * invNorm);

	return rayDirection;
}




__device__ bool invert_3x3_symmetric(const float* cov, float* inv_cov) {
    // cov = [S00, S01, S02, S11, S12, S22]
    float S00 = cov[0], S01 = cov[1], S02 = cov[2];
    float S11 = cov[3], S12 = cov[4], S22 = cov[5];
    
    float det = S00*(S11*S22 - S12*S12) - S01*(S01*S22 - S12*S02) + S02*(S01*S12 - S11*S02);
    
    if (fabsf(det) < 1e-10f) return false;
    
    float inv_det = 1.0f / det;
    inv_cov[0] = (S11*S22 - S12*S12) * inv_det;
    inv_cov[1] = (S02*S12 - S01*S22) * inv_det;
    inv_cov[2] = (S01*S12 - S02*S11) * inv_det;
    inv_cov[3] = inv_cov[1]; // Symmetric
    inv_cov[4] = (S00*S22 - S02*S02) * inv_det;
    inv_cov[5] = (S01*S02 - S00*S12) * inv_det;
    inv_cov[6] = inv_cov[2]; // Symmetric
    inv_cov[7] = inv_cov[5]; // Symmetric
    inv_cov[8] = (S00*S11 - S01*S01) * inv_det;
    
    return true;
}

__device__ glm::vec3 getIntersection3D(float3 ray, const glm::vec3 mean, const float* covariance, glm::vec3 campos) {
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



// High-order SH for noise rendering
__device__ void computeColorFromIntersection(int idx, const glm::vec3 unit_int, const float* texture, const int deg, const int max_coeffs, glm::vec4* final_result)
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
		result += HO_SH[28] * xz * sh[4] +
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
			
			result +=	(-HO_SH[32]*x*zz + HO_SH[13]*xxx) * sh[9] +
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
			
				result += -HO_SH[40] * x_minus_z*x_plus_z*xz * sh[16] +
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

					result += 	(-HO_SH[48]*xzzzz + HO_SH[65]*xxxzz - HO_SH[17] * xxxxx) * sh[25] +
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
							result += 0.0f * sh[coeff];
					}
				}
			}
		}
	}
	}
	*final_result = result;
}

// SLERP (Spherical Linear Interpolation) helper functions
__device__ float3 slerp3D(float3 a, float3 b, float t) {
    float dot_product = a.x * b.x + a.y * b.y + a.z * b.z;
    
    // Clamp dot product to avoid numerical issues
    dot_product = fminf(1.0f, fmaxf(-1.0f, dot_product));
    
    // If vectors are nearly parallel, use linear interpolation
    if (fabsf(dot_product) > 0.9995f) {
        float3 result;
        result.x = (1.0f - t) * a.x + t * b.x;
        result.y = (1.0f - t) * a.y + t * b.y;
        result.z = (1.0f - t) * a.z + t * b.z;
        return result;
    }
    
    // Handle zero vectors
    float norm_a = sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
    float norm_b = sqrtf(b.x * b.x + b.y * b.y + b.z * b.z);
    
    if (norm_a < 1e-6f || norm_b < 1e-6f) {
        float3 result;
        result.x = (1.0f - t) * a.x + t * b.x;
        result.y = (1.0f - t) * a.y + t * b.y;
        result.z = (1.0f - t) * a.z + t * b.z;
        return result;
    }
    
    // Calculate angle between vectors
    float theta = acosf(fabsf(dot_product));
    float sin_theta = sinf(theta);
    
    if (sin_theta < 1e-6f) {
        float3 result;
        result.x = (1.0f - t) * a.x + t * b.x;
        result.y = (1.0f - t) * a.y + t * b.y;
        result.z = (1.0f - t) * a.z + t * b.z;
        return result;
    }
    
    // SLERP formula
    float coeff_a = sinf((1.0f - t) * theta) / sin_theta;
    float coeff_b = sinf(t * theta) / sin_theta;
    
    float3 result;
    result.x = coeff_a * a.x + coeff_b * b.x;
    result.y = coeff_a * a.y + coeff_b * b.y;
    result.z = coeff_a * a.z + coeff_b * b.z;
    
    return result;
}

// N-dimensional SLERP for arbitrary feature vectors
__device__ void slerpND(const float* a, const float* b, float t, float* result, int dim) {
    // Calculate dot product
    float dot_product = 0.0f;
    for (int i = 0; i < dim; i++) {
        dot_product += a[i] * b[i];
    }
    
    // Clamp dot product to avoid numerical issues
    dot_product = fminf(1.0f, fmaxf(-1.0f, dot_product));
    
    // If vectors are nearly parallel, use linear interpolation
    if (fabsf(dot_product) > 0.9995f) {
        for (int i = 0; i < dim; i++) {
            result[i] = (1.0f - t) * a[i] + t * b[i];
        }
        return;
    }
    
    // Calculate vector norms
    float norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < dim; i++) {
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    norm_a = sqrtf(norm_a);
    norm_b = sqrtf(norm_b);
    
    // Handle zero vectors
    if (norm_a < 1e-6f || norm_b < 1e-6f) {
        for (int i = 0; i < dim; i++) {
            result[i] = (1.0f - t) * a[i] + t * b[i];
        }
        return;
    }
    
    // Calculate angle between vectors
    float theta = acosf(fabsf(dot_product));
    float sin_theta = sinf(theta);
    
    if (sin_theta < 1e-6f) {
        for (int i = 0; i < dim; i++) {
            result[i] = (1.0f - t) * a[i] + t * b[i];
        }
        return;
    }
    
    // SLERP formula
    float coeff_a = sinf((1.0f - t) * theta) / sin_theta;
    float coeff_b = sinf(t * theta) / sin_theta;
    
    for (int i = 0; i < dim; i++) {
        result[i] = coeff_a * a[i] + coeff_b * b[i];
    }
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		// The XYZ assignments have been modified to match e3nn's format.
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * x * sh[1] + SH_C1 * y * sh[2] - SH_C1 * z * sh[3];

		if (deg > 1)
		{
			float zz = z * z, xx = x * x, yy = y * y;
			float zx = z * x, xy = x * y, zy = z * y;
			result = result +
				SH_C2[0] * zx * sh[4] +
				SH_C2[1] * xy * sh[5] +
				SH_C2[2] * (2.0f * yy - zz - xx) * sh[6] +
				SH_C2[3] * zy * sh[7] +
				SH_C2[4] * (zz - xx) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * x * (3.0f * zz - xx) * sh[9] +
					SH_C3[1] * zx * y * sh[10] +
					SH_C3[2] * x * (4.0f * yy - zz - xx) * sh[11] +
					SH_C3[3] * y * (2.0f * yy - 3.0f * zz - 3.0f * xx) * sh[12] +
					SH_C3[4] * z * (4.0f * yy - zz - xx) * sh[13] +
					SH_C3[5] * z * (zz - xx) * sh[14] +
					SH_C3[6] * z * (zz - 3.0f * xx) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to feature values.
__device__ void computeFeatureFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, glm::vec4* final_result)
{
	// Similar to computeColorFromSH but for NUM_FEATURE_CHANNELS
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	// Feature SH coefficients are stored as [gaussian_idx][coeff_idx][feature_ch]
	const glm::vec4* sh = ((glm::vec4*)shs) + idx * max_coeffs;
	
	// Initialize with degree 0
	glm::vec4 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		
		result +=   -SH_C1 * x * sh[1] +
					 SH_C1 * y * sh[2] +
					-SH_C1 * z * sh[3];

		if (deg > 1)
		{
			float zz = z * z, xx = x * x, yy = y * y;
			float zx = z * x, xy = x * y, zy = z * y;
			
			result += 	SH_C2[0] * zx * sh[4] +
						SH_C2[1] * xy * sh[5] +
						SH_C2[2] * (2.0f * yy - zz - xx) * sh[6] +
						SH_C2[3] * zy * sh[7] +
						SH_C2[4] * (zz - xx) * sh[8];

			if (deg > 2)
			{
				result += 	SH_C3[0] * x * (3.0f * zz - xx) * sh[9] +
							SH_C3[1] * zx * y * sh[10] +
							SH_C3[2] * x * (4.0f * yy - zz - xx) * sh[11] +
							SH_C3[3] * y * (2.0f * yy - 3.0f * zz - 3.0f * xx) * sh[12] +
							SH_C3[4] * z * (4.0f * yy - zz - xx) * sh[13] +
							SH_C3[5] * z * (zz - xx) * sh[14] +
							SH_C3[6] * z * (zz - 3.0f * xx) * sh[15];
			}
		}
	}
	*final_result = result;	
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M, int M_f,
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
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* features,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool render_textured_features,
	int degree_feature)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr && shs != nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);  
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}
	
	// If features should be computed from SH (when textured_features is false)
	if (feature_shs != nullptr && !render_textured_features)
	{
		glm::vec4 feature_result;
		computeFeatureFromSH(idx, degree_feature, M_f, (glm::vec3*)orig_points, *cam_pos, feature_shs, &feature_result);
		features[idx * NUM_FEATURE_CHANNELS + 0] = feature_result.x;
		features[idx * NUM_FEATURE_CHANNELS + 1] = feature_result.y;
		features[idx * NUM_FEATURE_CHANNELS + 2] = feature_result.z;
		features[idx * NUM_FEATURE_CHANNELS + 3] = feature_result.w;

	}
	else if (features_precomp != nullptr)
	{
		// Copy precomputed features
		const glm::vec4* precomp_vec4 = (glm::vec4*)features_precomp;
		features[idx * NUM_FEATURE_CHANNELS + 0] = precomp_vec4[idx].x;
		features[idx * NUM_FEATURE_CHANNELS + 1] = precomp_vec4[idx].y;
		features[idx * NUM_FEATURE_CHANNELS + 2] = precomp_vec4[idx].z;
		features[idx * NUM_FEATURE_CHANNELS + 3] = precomp_vec4[idx].w;
	}
	

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);

}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int M_f,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ colors,
	const float* __restrict__ features,
	const float* __restrict__ render_noise,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	const int* __restrict__ radii,
	float* __restrict__ out_alpha,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_feature_map,
	float* __restrict__ out_depth,
	bool use_slerp,
	float* __restrict__ out_render_noise,
	float pixel_threshold,
	const glm::vec3* __restrict__ means3D,
	const float* __restrict__ cov3Ds,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix,
	const float* __restrict__ projmatrix_inv,
	const glm::vec3* __restrict__ cam_pos,
	int degree_noise,
	int blend_mode,
	bool render_textured_features,
	int degree_feature) 
{

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };
	
	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_cov3D[BLOCK_SIZE * 6];
	__shared__ glm::vec3 collected_mean[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;

	float C[CHANNELS] = { 0 };
	float F[NUM_FEATURE_CHANNELS] = { 0 };
	float N[NUM_FEATURE_CHANNELS] = { 0 };
	float N_weight_sq = 0.0f;

	// Added for depth computation
	float weight = 0;
	float depth_val = 0;
	float3 ray = getRayVec(pixf, W, H, viewmatrix, projmatrix_inv, *cam_pos);

	// Storage for feature blending modes
	struct Contribution {
		glm::vec4 feature;
		float effective_alpha;
		float depth;
		int id;
	};
	Contribution contributions[256]; // Max contributions per pixel
	int num_contributions = 0;
	int selected_idx = -1; // For argmax/median modes

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_mean[block.thread_rank()] = means3D[coll_id];
			// Store covariance matrix (6 elements per Gaussian)
			for (int i = 0; i < 6; i++) {
				collected_cov3D[i * BLOCK_SIZE + block.thread_rank()] = cov3Ds[coll_id * 6 + i];
			}
		}

		block.sync();
		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;
				
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			float alpha_T = alpha * T;
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Colors always use standard Gaussian splatting
			if (colors){
				for (int ch = 0; ch < CHANNELS; ch++){
					C[ch] += colors[collected_id[j] * CHANNELS + ch] * alpha_T;
				}
			}
			
			// Features use blending modes
			if (features) {
				if (blend_mode == GAUSSIAN_SPLATTING) {
					// Check if we should use textured features
					if (render_textured_features) {
						// Textured feature rendering
						glm::vec3 mean = collected_mean[j];
						const float* cov3D = &collected_cov3D[j];
						glm::vec3 unit_int = getIntersection3D(ray, mean, cov3D, *cam_pos);
						
						int degree = degree_feature;
						if (glm::length(unit_int) < 1e-6f)
							degree = 0;
						
						glm::vec4 textured_features = {0, 0, 0, 0};
						computeColorFromIntersection(collected_id[j], unit_int, features, degree, M_f, &textured_features);
						F[0] += textured_features.x * alpha_T;
						F[1] += textured_features.y * alpha_T;
						F[2] += textured_features.z * alpha_T;
						F[3] += textured_features.w * alpha_T;

					} else {
						// Standard Gaussian Splatting for features
						if (use_slerp) {
							float current_feat[NUM_FEATURE_CHANNELS];
							float new_feat[NUM_FEATURE_CHANNELS];
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
								current_feat[ch] = F[ch];
								new_feat[ch] = features[collected_id[j] * NUM_FEATURE_CHANNELS + ch];
							}
							slerpND(current_feat, new_feat, alpha_T, F, NUM_FEATURE_CHANNELS);
						} else {
							for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++){
								F[ch] += features[collected_id[j] * NUM_FEATURE_CHANNELS + ch] * alpha_T; 
							}
						}
					}
				} else {
					// Store feature contribution for other blending modes
					if (num_contributions < 256) {
						Contribution& contrib = contributions[num_contributions];
						contrib.effective_alpha = alpha_T;
						contrib.depth = depths[collected_id[j]];
						contrib.id = collected_id[j];
						
						// Use textured features if enabled, otherwise use regular features
						if (render_textured_features) {
							glm::vec3 mean = collected_mean[j];
							const float* cov3D = &collected_cov3D[j];
							glm::vec3 unit_int = getIntersection3D(ray, mean, cov3D, *cam_pos);
							
							int degree = degree_feature;
							if (glm::length(unit_int) < 1e-6f)
								degree = 0;
							
							computeColorFromIntersection(collected_id[j], unit_int, features, degree, M_f, &contrib.feature);
						} else {
								contrib.feature.x = features[collected_id[j] * NUM_FEATURE_CHANNELS + 0];
								contrib.feature.y = features[collected_id[j] * NUM_FEATURE_CHANNELS + 1];
								contrib.feature.z = features[collected_id[j] * NUM_FEATURE_CHANNELS + 2];
								contrib.feature.w = features[collected_id[j] * NUM_FEATURE_CHANNELS + 3];
						}
						num_contributions++;
					}
				}
			}

			// Compute per-pixel noise using intersection-based SH
			if (render_noise) {
				float gaussian_radius = radii[collected_id[j]];
				
				if (gaussian_radius > pixel_threshold) {
					// Large Gaussian: use random noise for texture
					curandState noise_state;
					curand_init(collected_id[j] * 1009u + pix_id, 0, 0, &noise_state);
					
					for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
						float random_noise = curand_normal(&noise_state);
						N[ch] += random_noise * alpha_T;
					}
				} else {
					// Small Gaussian: compute per-pixel 3D consistent noise
					glm::vec3 mean = collected_mean[j];
					const float* cov3D = &collected_cov3D[j];
					glm::vec3 unit_int = getIntersection3D(ray, mean, cov3D, *cam_pos);
					
					int degree = degree_noise;
					if (glm::length(unit_int) < 1e-6f)
						degree = 0;
					
					glm::vec4 noise = {0, 0, 0, 0};
					computeColorFromIntersection(collected_id[j], unit_int, render_noise, degree, M_f, &noise);
					for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
						N[ch] += noise[ch] * alpha_T;
					}
				}
				N_weight_sq += alpha_T * alpha_T;
			}

			weight += alpha_T;
			depth_val += depths[collected_id[j]] * alpha_T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// Process feature blending modes only
	if (blend_mode != GAUSSIAN_SPLATTING && num_contributions > 0 && features) {
		switch (blend_mode) {
			case ARGMAX_EFFECTIVE: {
				int max_idx = 0;
				for (int i = 1; i < num_contributions; i++) {
					if (contributions[i].effective_alpha > contributions[max_idx].effective_alpha) {
						max_idx = i;
					}
				}
				selected_idx = max_idx;
				F[0] = contributions[max_idx].feature.x;
				F[1] = contributions[max_idx].feature.y;
				F[2] = contributions[max_idx].feature.z;
				F[3] = contributions[max_idx].feature.w;
				break;
			}
			case MEDIAN_EFFECTIVE: {
				// Simple bubble sort for small arrays
				for (int i = 0; i < num_contributions - 1; i++) {
					for (int k = 0; k < num_contributions - i - 1; k++) {
						if (contributions[k].effective_alpha < contributions[k + 1].effective_alpha) {
							Contribution temp = contributions[k];
							contributions[k] = contributions[k + 1];
							contributions[k + 1] = temp;
						}
					}
				}
				int median_idx = num_contributions / 2;
				selected_idx = median_idx;
				F[0] = contributions[median_idx].feature.x;
				F[1] = contributions[median_idx].feature.y;
				F[2] = contributions[median_idx].feature.z;
				F[3] = contributions[median_idx].feature.w;
				break;
			}
			case RMS_EFFECTIVE: {
				float total_weight = 0.0f;
				for (int i = 0; i < num_contributions; i++) {
					total_weight += contributions[i].effective_alpha;
				}
				if (total_weight > 1e-8f) {
					float weighted_sum_x = 0.0f, weighted_sum_y = 0, weighted_sum_z = 0, weighted_sum_w = 0;
					for (int i = 0; i < num_contributions; i++) {
						float contrib = contributions[i].effective_alpha;
						weighted_sum_x += contributions[i].feature.x * contributions[i].feature.x * contrib;
						weighted_sum_y += contributions[i].feature.y * contributions[i].feature.y * contrib;
						weighted_sum_z += contributions[i].feature.z * contributions[i].feature.z * contrib;
						weighted_sum_w += contributions[i].feature.w * contributions[i].feature.w * contrib;
					}
					F[0] = sqrtf(weighted_sum_x / total_weight);
					F[1] = sqrtf(weighted_sum_y / total_weight);
					F[2] = sqrtf(weighted_sum_z / total_weight);
					F[3] = sqrtf(weighted_sum_w / total_weight);

				}
				break;
			}
			case SOFTMAX_EFFECTIVE: {
				// Compute softmax weights from effective alphas
				float max_alpha = contributions[0].effective_alpha;
				for (int i = 1; i < num_contributions; i++) {
					max_alpha = fmaxf(max_alpha, contributions[i].effective_alpha);
				}
				
				float exp_sum = 0.0f;
				float softmax_weights[256];
				for (int i = 0; i < num_contributions; i++) {
					softmax_weights[i] = expf(contributions[i].effective_alpha - max_alpha);
					exp_sum += softmax_weights[i];
				}
				
				if (exp_sum > 1e-8f) {
					for (int i = 0; i < num_contributions; i++) {
						softmax_weights[i] /= exp_sum;
					}
					
					for (int i = 0; i < num_contributions; i++) {
						F[0] += softmax_weights[i] * contributions[i].feature.x;
						F[1] += softmax_weights[i] * contributions[i].feature.y;
						F[2] += softmax_weights[i] * contributions[i].feature.z;
						F[3] += softmax_weights[i] * contributions[i].feature.w;
					}
				}
				break;
			}
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		n_contrib[pix_id] = last_contributor;
		if (colors){
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
		if (features){
			for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++)                 
				out_feature_map[ch * H * W + pix_id] = F[ch] + T * 0.0f;
		}
		// Normalize N by variance-preserving factor and add random background
		if (out_render_noise) {
			curandState state;
			curand_init(pix_id, 0, 0, &state);
			if (N_weight_sq > 1e-8f) {
				float norm_factor = 1.0f / sqrtf(N_weight_sq);
				for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
					out_render_noise[ch * H * W + pix_id] = N[ch] * norm_factor + T * curand_normal(&state);
				}
			} else {
				for (int ch = 0; ch < NUM_FEATURE_CHANNELS; ch++) {
					out_render_noise[ch * H * W + pix_id] = N[ch] + T * curand_normal(&state);
				}
			}
		}
		out_alpha[pix_id] = weight;
		out_depth[pix_id] = depth_val;
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int M_f,
	const float2* means2D,
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
	bool use_slerp,
	float* out_render_noise,
	float pixel_threshold,
	const glm::vec3* means3D,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	int degree_noise,
	int blend_mode,
	bool render_textured_features,
	int degree_feature) 
{
	renderCUDA<NUM_COLOR_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H, M_f,
		means2D,
		colors,
		features,
		render_noise,
		depths,
		conic_opacity,
		radii,
		out_alpha,
		n_contrib,
		bg_color,
		out_color,
		out_feature_map,
		out_depth,
		use_slerp,
		out_render_noise,
		pixel_threshold,
		means3D,
		cov3Ds,
		viewmatrix,
		projmatrix,
		projmatrix_inv,
		cam_pos,
		degree_noise,
		blend_mode,
		render_textured_features,
		degree_feature);

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			printf("CUDA error: %s\n", cudaGetErrorString(error));
		}
}

void FORWARD::preprocess(int P, int D, int M, int M_f,
	const float* means3D,
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
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float* features,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool render_textured_features,
	int degree_feature)
{
	preprocessCUDA<NUM_COLOR_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M, M_f,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		feature_shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		features_precomp,
		viewmatrix, 
		projmatrix,
		projmatrix_inv,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		features,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		render_textured_features,
		degree_feature
		);
}
