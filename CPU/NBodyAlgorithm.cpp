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
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A t�megbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig �sszeszorozni
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

void NBodyAlgorithm::calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, float3(&accI)[4]) {
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

void NBodyAlgorithm::calculateAccelerationWithColor(const float3(&posI)[4], const float massJ, const float3 posJ, float3(&accI)[4], unsigned int(&isClose)[4]) {
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
    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 cmpDistance = _mm_set_ps1(float(mp_properties->positionScale));
    __m128 close = _mm_cmple_ps(rabs, cmpDistance);

    for (int i = 0; i < 4; i++) {
        if (close.m128_f32[i] == 0) {
            isClose[3 - i] = 0;
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



void NBodyAlgorithm::calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, float* accI) {
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
    __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

    __m128 m = _mm_set_ps1(massJ);
    __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

    __m128 aix = _mm_mul_ps(rx, rabsInv);
    __m128 aiy = _mm_mul_ps(ry, rabsInv);
    __m128 aiz = _mm_mul_ps(rz, rabsInv);

    //TODO KICSOMAGOLNI SIMA FLOATBA
    _mm_storer_ps(accI, aix);
    _mm_storer_ps(accI + 4, aiy);
    _mm_storer_ps(accI + 8, aiz);
}


void NBodyAlgorithm::calculateAcceleration(const float3(&posI)[4], const float massJ, const float3 posJ, __m128 accIx, __m128 accIy, __m128 accIz, float *accI) {
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
}
