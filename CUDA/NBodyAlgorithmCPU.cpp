#include "NBodyAlgorithmCPU.h"
#include "NBodyProperties.h"
#include <math.h>

float3 NBodyAlgorithmCPU::calculateAcceleration(const float3 posI, const float massJ, const float3 posJ) {
    float3 r, accI;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + mp_properties->EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
    accI.x = r.x * temp;
    accI.y = r.y * temp;
    accI.z = r.z * temp;
    return accI;
}

float3 NBodyAlgorithmCPU::calculateAccelerationWithColor(const float3 posI, const float massJ, const float3 posJ, unsigned int &numNeighbours) {
    float3 r, accI;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + mp_properties->EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;
    
    numNeighbours = (rabs < (float)mp_properties->positionScale) ? numNeighbours + 1 : numNeighbours;

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindi  g összeszorozni
    accI.x = r.x * temp;
    accI.y = r.y * temp;
    accI.z = r.z * temp;
    return accI;
}

/*void NBodyAlgorithm::calculateAcceleration(const float3 (&posI)[4], const float massJ, const float3 posJ, float3 (&accI)[4]) {
__declspec(align(16)) __m128 pix = _mm_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x);
__declspec(align(16)) __m128 piy = _mm_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y);
__declspec(align(16)) __m128 piz = _mm_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z);

__declspec(align(16)) __m128 pjx = _mm_set_ps1(posJ.x);
__declspec(align(16)) __m128 pjy = _mm_set_ps1(posJ.y);
__declspec(align(16)) __m128 pjz = _mm_set_ps1(posJ.z);

__declspec(align(16)) __m128 rx = _mm_sub_ps(pjx, pix);
__declspec(align(16)) __m128 ry = _mm_sub_ps(pjy, piy);
__declspec(align(16)) __m128 rz = _mm_sub_ps(pjz, piz);

__declspec(align(16)) __m128 eps2 = _mm_set_ps1(mp_properties->eps2);

__declspec(align(16)) __m128 rx2 = _mm_mul_ps(rx, rx);
__declspec(align(16)) __m128 ry2 = _mm_mul_ps(ry, ry);
__declspec(align(16)) __m128 rz2 = _mm_mul_ps(rz, rz);
//__m128 rabs = _mm_rsqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

//__m128 m = _mm_set_ps(massJ, massJ, massJ, massJ);
//__m128 rabsInv = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs), m);

__declspec(align(16)) __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

__declspec(align(16)) __m128 m = _mm_set_ps1(massJ);
__declspec(align(16)) __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

__declspec(align(16)) __m128 aix = _mm_mul_ps(rx, rabsInv);
__declspec(align(16)) __m128 aiy = _mm_mul_ps(ry, rabsInv);
__declspec(align(16)) __m128 aiz = _mm_mul_ps(rz, rabsInv);

for (int i = 0; i < 4; i++) {
accI[3-i].x = aix.m128_f32[i];
accI[3-i].y = aiy.m128_f32[i];
accI[3-i].z = aiz.m128_f32[i];
}

}*/

void NBodyAlgorithmCPU::calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, float3(&accI)[4]) {
    __m128 pix = _mm_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x);
    __m128 piy = _mm_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y);
    __m128 piz = _mm_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z);

    __m128 pjx = _mm_set_ps1(posJ.x);
    __m128 pjy = _mm_set_ps1(posJ.y);
    __m128 pjz = _mm_set_ps1(posJ.z);

    __m128 rx = _mm_sub_ps(pjx, pix);
    __m128 ry = _mm_sub_ps(pjy, piy);
    __m128 rz = _mm_sub_ps(pjz, piz);

    __m128 eps2 = _mm_set_ps1(mp_properties->EPS2);

    __m128 rx2 = _mm_mul_ps(rx, rx);
    __m128 ry2 = _mm_mul_ps(ry, ry);
    __m128 rz2 = _mm_mul_ps(rz, rz);
    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 m = _mm_set_ps1(massJ);
    __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

    __m128 aix = _mm_mul_ps(rx, rabsInv);
    __m128 aiy = _mm_mul_ps(ry, rabsInv);
    __m128 aiz = _mm_mul_ps(rz, rabsInv);

    for (int i = 0; i < 4; i++) {
        accI[3 - i].x = aix.m128_f32[i];
        accI[3 - i].y = aiy.m128_f32[i];
        accI[3 - i].z = aiz.m128_f32[i];
    }

}

void NBodyAlgorithmCPU::calculateAccelerationWithColor(const float3(&posI)[4], const float massJ, const float3 posJ, float3(&accI)[4], unsigned int(&numNeighbours)[4]) {
    __m128 pix = _mm_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x);
    __m128 piy = _mm_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y);
    __m128 piz = _mm_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z);

    __m128 pjx = _mm_set_ps1(posJ.x);
    __m128 pjy = _mm_set_ps1(posJ.y);
    __m128 pjz = _mm_set_ps1(posJ.z);

    __m128 rx = _mm_sub_ps(pjx, pix);
    __m128 ry = _mm_sub_ps(pjy, piy);
    __m128 rz = _mm_sub_ps(pjz, piz);

    __m128 eps2 = _mm_set_ps1(mp_properties->EPS2);

    __m128 rx2 = _mm_mul_ps(rx, rx);
    __m128 ry2 = _mm_mul_ps(ry, ry);
    __m128 rz2 = _mm_mul_ps(rz, rz);
    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 cmpDistance = _mm_set_ps1(float(mp_properties->positionScale));
    __m128 close = _mm_cmple_ps(rabs, cmpDistance);

    for (int i = 0; i < 4; i++) {
        if (close.m128_f32[i] == 0) {
            numNeighbours[3 - i] = 0;
        }
    }

    __m128 m = _mm_set_ps1(massJ);
    __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

    __m128 aix = _mm_mul_ps(rx, rabsInv);
    __m128 aiy = _mm_mul_ps(ry, rabsInv);
    __m128 aiz = _mm_mul_ps(rz, rabsInv);

    for (int i = 0; i < 4; i++) {
        accI[3 - i].x = aix.m128_f32[i];
        accI[3 - i].y = aiy.m128_f32[i];
        accI[3 - i].z = aiz.m128_f32[i];
    }

}


void NBodyAlgorithmCPU::calculateAcceleration(const float3(&posI)[8], const float massJ, const float3 posJ, float3(&accI)[8]) {
    __m256 pix = _mm256_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x, posI[4].x, posI[5].x, posI[6].x, posI[7].x);
    __m256 piy = _mm256_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y, posI[4].y, posI[5].y, posI[6].y, posI[7].y);
    __m256 piz = _mm256_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z, posI[4].z, posI[5].z, posI[6].z, posI[7].z);

    __m256 pjx = _mm256_set1_ps(posJ.x);
    __m256 pjy = _mm256_set1_ps(posJ.y);
    __m256 pjz = _mm256_set1_ps(posJ.z);

    __m256 rx = _mm256_sub_ps(pjx, pix);
    __m256 ry = _mm256_sub_ps(pjy, piy);
    __m256 rz = _mm256_sub_ps(pjz, piz);

    __m256 eps2 = _mm256_set1_ps(mp_properties->EPS2);

    __m256 rx2 = _mm256_mul_ps(rx, rx);
    __m256 ry2 = _mm256_mul_ps(ry, ry);
    __m256 rz2 = _mm256_mul_ps(rz, rz);
    __m256 rabs = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(rx2, ry2), _mm256_add_ps(rz2, eps2)));

    __m256 m = _mm256_set1_ps(massJ);
    __m256 rabsInv = _mm256_div_ps(m, _mm256_mul_ps(_mm256_mul_ps(rabs, rabs), rabs));

    __m256 aix = _mm256_mul_ps(rx, rabsInv);
    __m256 aiy = _mm256_mul_ps(ry, rabsInv);
    __m256 aiz = _mm256_mul_ps(rz, rabsInv);

    for (int i = 0; i < 8; i++) {
        accI[7 - i].x = aix.m256_f32[i];
        accI[7 - i].y = aiy.m256_f32[i];
        accI[7 - i].z = aiz.m256_f32[i];
    }

}

void NBodyAlgorithmCPU::calculateAccelerationWithColor(const float3(&posI)[8], const float massJ, const float3 posJ, float3(&accI)[8], unsigned int(&numNeighbours)[8]) {
    __m256 pix = _mm256_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x, posI[4].x, posI[5].x, posI[6].x, posI[7].x);
    __m256 piy = _mm256_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y, posI[4].y, posI[5].y, posI[6].y, posI[7].y);
    __m256 piz = _mm256_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z, posI[4].z, posI[5].z, posI[6].z, posI[7].z);

    __m256 pjx = _mm256_set1_ps(posJ.x);
    __m256 pjy = _mm256_set1_ps(posJ.y);
    __m256 pjz = _mm256_set1_ps(posJ.z);

    __m256 rx = _mm256_sub_ps(pjx, pix);
    __m256 ry = _mm256_sub_ps(pjy, piy);
    __m256 rz = _mm256_sub_ps(pjz, piz);

    __m256 eps2 = _mm256_set1_ps(mp_properties->EPS2);

    __m256 rx2 = _mm256_mul_ps(rx, rx);
    __m256 ry2 = _mm256_mul_ps(ry, ry);
    __m256 rz2 = _mm256_mul_ps(rz, rz);
    __m256 rabs = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(rx2, ry2), _mm256_add_ps(rz2, eps2)));

    __m256 cmpDistance = _mm256_set1_ps(float(mp_properties->positionScale));
    __m256 close = _mm256_cmp_ps(rabs, cmpDistance, 2);

    for (int i = 0; i < 8; i++) {
        if (close.m256_f32[i] == 0) {
            numNeighbours[7 - i] = 0;
        }
    }

    __m256 m = _mm256_set1_ps(massJ);
    __m256 rabsInv = _mm256_div_ps(m, _mm256_mul_ps(_mm256_mul_ps(rabs, rabs), rabs));

    __m256 aix = _mm256_mul_ps(rx, rabsInv);
    __m256 aiy = _mm256_mul_ps(ry, rabsInv);
    __m256 aiz = _mm256_mul_ps(rz, rabsInv);

    for (int i = 0; i < 8; i++) {
        accI[7 - i].x = aix.m256_f32[i];
        accI[7 - i].y = aiy.m256_f32[i];
        accI[7 - i].z = aiz.m256_f32[i];
    }

}

void NBodyAlgorithmCPU::calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, float *accI) {
    __m128 pix = _mm_set_ps(posI[3].x, posI[2].x, posI[1].x, posI[0].x);
    __m128 piy = _mm_set_ps(posI[3].y, posI[2].y, posI[1].y, posI[0].y);
    __m128 piz = _mm_set_ps(posI[3].z, posI[2].z, posI[1].z, posI[0].z);

    __m128 pjx = _mm_set_ps1(posJ.x);
    __m128 pjy = _mm_set_ps1(posJ.y);
    __m128 pjz = _mm_set_ps1(posJ.z);

    __m128 rx = _mm_sub_ps(pjx, pix);
    __m128 ry = _mm_sub_ps(pjy, piy);
    __m128 rz = _mm_sub_ps(pjz, piz);

    __m128 eps2 = _mm_set_ps1(mp_properties->EPS2);

    __m128 rx2 = _mm_mul_ps(rx, rx);
    __m128 ry2 = _mm_mul_ps(ry, ry);
    __m128 rz2 = _mm_mul_ps(rz, rz);
    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 m = _mm_set_ps1(massJ);
    __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

    __m128 aix = _mm_mul_ps(rx, rabsInv);
    __m128 aiy = _mm_mul_ps(ry, rabsInv);
    __m128 aiz = _mm_mul_ps(rz, rabsInv);

    _mm_store_ps(accI, aix);
    _mm_store_ps(accI + 4, aiy);
    _mm_store_ps(accI + 8, aiz);
}

void NBodyAlgorithmCPU::calculateAcceleration(const float3(&posI)[8], const float massJ, const float3 posJ, float *accI) {
    __m256 pix = _mm256_set_ps(posI[7].x, posI[6].x, posI[5].x, posI[4].x, posI[3].x, posI[2].x, posI[1].x, posI[0].x);
    __m256 piy = _mm256_set_ps(posI[7].y, posI[6].y, posI[5].y, posI[4].y, posI[3].y, posI[2].y, posI[1].y, posI[0].y);
    __m256 piz = _mm256_set_ps(posI[7].z, posI[6].z, posI[5].z, posI[4].z, posI[3].z, posI[2].z, posI[1].z, posI[0].z);

    __m256 pjx = _mm256_set1_ps(posJ.x);
    __m256 pjy = _mm256_set1_ps(posJ.y);
    __m256 pjz = _mm256_set1_ps(posJ.z);

    __m256 rx = _mm256_sub_ps(pjx, pix);
    __m256 ry = _mm256_sub_ps(pjy, piy);
    __m256 rz = _mm256_sub_ps(pjz, piz);

    __m256 eps2 = _mm256_set1_ps(mp_properties->EPS2);

    __m256 rx2 = _mm256_mul_ps(rx, rx);
    __m256 ry2 = _mm256_mul_ps(ry, ry);
    __m256 rz2 = _mm256_mul_ps(rz, rz);
    __m256 rabs = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(rx2, ry2), _mm256_add_ps(rz2, eps2)));

    __m256 m = _mm256_set1_ps(massJ);
    __m256 rabsInv = _mm256_div_ps(m, _mm256_mul_ps(_mm256_mul_ps(rabs, rabs), rabs));

    __m256 aix = _mm256_mul_ps(rx, rabsInv);
    __m256 aiy = _mm256_mul_ps(ry, rabsInv);
    __m256 aiz = _mm256_mul_ps(rz, rabsInv);

    _mm256_store_ps(accI, aix);
    _mm256_store_ps(accI + 8, aiy);
    _mm256_store_ps(accI + 16, aiz);

}

/*void NBodyAlgorithmCPU::calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, __m128 accIx, __m128 accIy, __m128 accIz, float *accI) {
    __m128 pix = _mm_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x);
    __m128 piy = _mm_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y);
    __m128 piz = _mm_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z);

    __m128 pjx = _mm_set_ps1(posJ.x);
    __m128 pjy = _mm_set_ps1(posJ.y);
    __m128 pjz = _mm_set_ps1(posJ.z);

    __m128 rx = _mm_sub_ps(pjx, pix);
    __m128 ry = _mm_sub_ps(pjy, piy);
    __m128 rz = _mm_sub_ps(pjz, piz);

    __m128 eps2 = _mm_set_ps1(mp_properties->EPS2);

    __m128 rx2 = _mm_mul_ps(rx, rx);
    __m128 ry2 = _mm_mul_ps(ry, ry);
    __m128 rz2 = _mm_mul_ps(rz, rz);
    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 m = _mm_set_ps1(massJ);
    __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

    __m128 aix = _mm_mul_ps(rx, rabsInv);
    __m128 aiy = _mm_mul_ps(ry, rabsInv);
    __m128 aiz = _mm_mul_ps(rz, rabsInv);

    accIx = _mm_add_ps(accIx, aix);
    accIy = _mm_add_ps(accIy, aiy);
    accIz = _mm_add_ps(accIz, aiz);

    _mm_storer_ps(accI, accIx);
    _mm_storer_ps(accI + 4, accIy);
    _mm_storer_ps(accI + 8, accIz);
}*/
