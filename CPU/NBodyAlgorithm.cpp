#include "NBodyAlgorithm.h"
#include "NBodyProperties.h"
#include <math.h>

float3 NBodyAlgorithm::calculateAcceleration(const float3 posI, const float massJ, const float3 posJ) {
    float3 r(0.0f, 0.0f, 0.0f);
    float3 accI(0.0f, 0.0f, 0.0f);

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + mp_properties->eps2);
    float rabsInv = 1.0 / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A tömegbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
    accI.x = r.x * temp;
    accI.y = r.y * temp;
    accI.z = r.z * temp;
    return accI;
}

void NBodyAlgorithm::calculateAcceleration(const float3 (&posI)[4], const float massJ, const float3 posJ, float3 (&accI)[4]) {
    __m128 pix = _mm_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x);
    __m128 piy = _mm_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y);
    __m128 piz = _mm_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z);

    __m128 pjx = _mm_set_ps1(posJ.x);
    __m128 pjy = _mm_set_ps1(posJ.y);
    __m128 pjz = _mm_set_ps1(posJ.z);

    __m128 rx = _mm_sub_ps(pjx, pix);
    __m128 ry = _mm_sub_ps(pjy, piy);
    __m128 rz = _mm_sub_ps(pjz, piz);

    __m128 eps2 = _mm_set_ps1(mp_properties->eps2);

    __m128 rx2 = _mm_mul_ps(rx, rx);
    __m128 ry2 = _mm_mul_ps(ry, ry);
    __m128 rz2 = _mm_mul_ps(rz, rz);
    /*__m128 rabs = _mm_rsqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 m = _mm_set_ps(massJ, massJ, massJ, massJ);
    __m128 rabsInv = _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs), m);*/

    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 m = _mm_set_ps1(massJ);
    __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));
 
    __m128 aix = _mm_mul_ps(rx, rabsInv);
    __m128 aiy = _mm_mul_ps(ry, rabsInv);
    __m128 aiz = _mm_mul_ps(rz, rabsInv);

    for (int i = 0; i < 4; i++) {
        accI[3-i].x = aix.m128_f32[i];
        accI[3-i].y = aiy.m128_f32[i];
        accI[3-i].z = aiz.m128_f32[i];
    }

}