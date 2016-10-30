#include <assert.h>
#include "NBodySystemFlat.h"

void NBodySystemFlat::init() {
    assert(mp_properties->numBody > 1);
    assert(mp_properties->massScale != 0);
    assert(mp_properties->positionScale != 0);
    assert(mp_properties->velocityScale != 0);
    assert(mp_properties->startTime < mp_properties->endTime);
    //assert(!m_systemInitialized);

    srand(mp_properties->seed);

    float3 pos;
    pos.x = 0.0f; pos.x = 0.0f; pos.x = 0.0f;

    float3 vel;
    vel.x = 0.0f; vel.x = 0.0f; vel.x = 0.0f;

    float disterr = 10e-6f;

    mp_mass         = reinterpret_cast<float*>(_aligned_malloc(sizeof(float)*mp_properties->numBody, 32));
    mp_position     = reinterpret_cast<float*>(_aligned_malloc(3 * sizeof(float)*mp_properties->numBody, 32));
    mp_velocity     = reinterpret_cast<float*>(_aligned_malloc(3 * sizeof(float)*mp_properties->numBody, 32));
    mp_acceleration = reinterpret_cast<float*>(_aligned_malloc(3 * sizeof(float)*mp_properties->numBody, 32));

    for (int i = 0; i < mp_properties->numBody;) {

        pos = mp_initializator->getNewPosition();

        // Ellenõrizzük, hogy ne kerüljön egymáshoz túl közel két test
        // Bizonyos beállítások alkalmazása mellett, elõfordulhat, hogy kifagy a rendszer
        bool occupied = false;
        for (int j = 0; j < i; j++) {
            occupied = (abs(mp_position[3 * j] - pos.x) < disterr) &&
                (abs(mp_position[3 * j + 1] - pos.y) < disterr) &&
                (abs(mp_position[3 * j + 2] - pos.z) < disterr);
            if (occupied) break;
        }
        if (occupied) continue;

        mp_position[3 * i] = pos.x;
        mp_position[3 * i + 1] = pos.y;
        mp_position[3 * i + 2] = pos.z;

        mp_mass[i] = mp_initializator->getNewMass();

        vel = mp_initializator->getNewVelocity();
        mp_velocity[3 * i] = vel.x;
        mp_velocity[3 * i + 1] = vel.y;
        mp_velocity[3 * i + 2] = vel.z;

        mp_acceleration[3 * i] = 0.0f;
        mp_acceleration[3 * i + 1] = 0.0f;
        mp_acceleration[3 * i + 2] = 0.0f;

        i++;
    }

    m_systemInitialized = true;

}

void NBodySystemFlat::integrate() {
    assert(m_systemInitialized);
    while (mp_properties->currentTime < mp_properties->endTime) {

#ifdef OPENMP
#pragma omp parallel for
#endif
        /////////////////////////////////////////////
       /* for (int i = 0; i < mp_properties->numBody; i++) {
            float r[3], rabs, rabsInv, temp;
            mp_acceleration[3 * i] = 0.0f;
            mp_acceleration[3 * i + 1] = 0.0f;
            mp_acceleration[3 * i + 2] = 0.0f;

            float3 acc(0.0f, 0.0f, 0.0f);
            for (int j = 0; j < mp_properties->numBody; j++) {
                // 17 FLOPS
                
                r[0] = mp_position[3 * j] - mp_position[3 * i];
                r[1] = mp_position[3 * j + 1] - mp_position[3 * i + 1];
                r[2] = mp_position[3 * j + 2] - mp_position[3 * i + 2];

                rabs = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + mp_properties->EPS2);
                rabsInv = 1.0f / (rabs * rabs * rabs);
                temp = mp_mass[j] * rabsInv;

                // A tömegbe bele van olvasztva a G
                // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
                // 3 FLOPS
                mp_acceleration[3 * i] += r[0] * temp;
                mp_acceleration[3 * i + 1] += r[1] * temp;
                mp_acceleration[3 * i + 2] += r[2] * temp;
                
            }
        }*/
        /////////////////////////////////////////////
        for (int i = 0; i < mp_properties->numBody; i += 4) {
            mp_acceleration[3 * i] = 0.0f;        mp_acceleration[3 * i + 1] = 0.0f;        mp_acceleration[3 * i + 2] = 0.0f;
            mp_acceleration[3 * (i + 1)] = 0.0f;  mp_acceleration[3 * (i + 1) + 1] = 0.0f;  mp_acceleration[3 * (i + 1) + 2] = 0.0f;
            mp_acceleration[3 * (i + 2)] = 0.0f;  mp_acceleration[3 * (i + 2) + 1] = 0.0f;  mp_acceleration[3 * (i + 2) + 2] = 0.0f;
            mp_acceleration[3 * (i + 3)] = 0.0f;  mp_acceleration[3 * (i + 3) + 1] = 0.0f;  mp_acceleration[3 * (i + 3) + 2] = 0.0f;

            for (int j = 0; j < mp_properties->numBody; j++) {

                __declspec(align(16)) __m128 pix = _mm_set_ps(mp_position[3 * i], mp_position[3 * (i + 1)], mp_position[3 * (i + 2)], mp_position[3 * (i + 3)]);
                __declspec(align(16)) __m128 piy = _mm_set_ps(mp_position[3 * (i)+1], mp_position[3 * (i + 1) + 1], mp_position[3 * (i + 2) + 1], mp_position[3 * (i + 3) + 1]);
                __declspec(align(16)) __m128 piz = _mm_set_ps(mp_position[3 * (i)+2], mp_position[3 * (i + 1) + 2], mp_position[3 * (i + 2) + 2], mp_position[3 * (i + 3) + 2]);

                __declspec(align(16)) __m128 pjx = _mm_set_ps1(mp_position[3 * j]);
                __declspec(align(16)) __m128 pjy = _mm_set_ps1(mp_position[3 * j + 1]);
                __declspec(align(16)) __m128 pjz = _mm_set_ps1(mp_position[3 * j + 2]);

                __declspec(align(16)) __m128 rx = _mm_sub_ps(pjx, pix);
                __declspec(align(16)) __m128 ry = _mm_sub_ps(pjy, piy);
                __declspec(align(16)) __m128 rz = _mm_sub_ps(pjz, piz);

                __declspec(align(16)) __m128 eps2 = _mm_set_ps1(mp_properties->EPS2);

                __declspec(align(16)) __m128 rx2 = _mm_mul_ps(rx, rx);
                __declspec(align(16)) __m128 ry2 = _mm_mul_ps(ry, ry);
                __declspec(align(16)) __m128 rz2 = _mm_mul_ps(rz, rz);
                __declspec(align(16)) __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

                __declspec(align(16)) __m128 m = _mm_set_ps1(mp_mass[j]);
                __declspec(align(16)) __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

                __declspec(align(16)) __m128 aix = _mm_mul_ps(rx, rabsInv);
                __declspec(align(16)) __m128 aiy = _mm_mul_ps(ry, rabsInv);
                __declspec(align(16)) __m128 aiz = _mm_mul_ps(rz, rabsInv);

                for (int k = 0; k < 4; k++) {
                    mp_acceleration[(3 * (i + 3 - k))] = aix.m128_f32[k];
                    mp_acceleration[(3 * (i + 3 - k)) + 1] = aiy.m128_f32[k];
                    mp_acceleration[(3 * (i + 3 - k)) + 2] = aiz.m128_f32[k];
                }
            }
        }
        /////////////////////////////////////////////
        /*for (int i = 0; i < mp_properties->numBody; i += 8) {

            mp_acceleration[3 * i] = 0.0f;        mp_acceleration[3 * i + 1] = 0.0f;        mp_acceleration[3 * i + 2] = 0.0f;
            mp_acceleration[3 * (i + 1)] = 0.0f;  mp_acceleration[3 * (i + 1) + 1] = 0.0f;  mp_acceleration[3 * (i + 1) + 2] = 0.0f;
            mp_acceleration[3 * (i + 2)] = 0.0f;  mp_acceleration[3 * (i + 2) + 1] = 0.0f;  mp_acceleration[3 * (i + 2) + 2] = 0.0f;
            mp_acceleration[3 * (i + 3)] = 0.0f;  mp_acceleration[3 * (i + 3) + 1] = 0.0f;  mp_acceleration[3 * (i + 3) + 2] = 0.0f;
            mp_acceleration[3 * (i + 4)] = 0.0f;  mp_acceleration[3 * (i + 4) + 1] = 0.0f;  mp_acceleration[3 * (i + 4) + 2] = 0.0f;
            mp_acceleration[3 * (i + 5)] = 0.0f;  mp_acceleration[3 * (i + 5) + 1] = 0.0f;  mp_acceleration[3 * (i + 5) + 2] = 0.0f;
            mp_acceleration[3 * (i + 6)] = 0.0f;  mp_acceleration[3 * (i + 6) + 1] = 0.0f;  mp_acceleration[3 * (i + 6) + 2] = 0.0f;
            mp_acceleration[3 * (i + 7)] = 0.0f;  mp_acceleration[3 * (i + 7) + 1] = 0.0f;  mp_acceleration[3 * (i + 7) + 2] = 0.0f;

            for (int j = 0; j < mp_properties->numBody; j++) {

                __m256 pix = _mm256_set_ps(mp_position[3 * i], mp_position[3 * (i + 1)], mp_position[3 * (i + 2)], mp_position[3 * (i + 3)], mp_position[3 * (i + 4)], mp_position[3 * (i + 5)], mp_position[3 * (i + 6)], mp_position[3 * (i + 7)]);
                __m256 piy = _mm256_set_ps(mp_position[3 * i + 1], mp_position[3 * (i + 1) + 1], mp_position[3 * (i + 2) + 1], mp_position[3 * (i + 3) + 1], mp_position[3 * (i + 4) + 1], mp_position[3 * (i + 5) + 1], mp_position[3 * (i + 6) + 1], mp_position[3 * (i + 7) + 1]);
                __m256 piz = _mm256_set_ps(mp_position[3 * i + 2], mp_position[3 * (i + 1) + 2], mp_position[3 * (i + 2) + 2], mp_position[3 * (i + 3) + 2], mp_position[3 * (i + 4) + 2], mp_position[3 * (i + 5) + 2], mp_position[3 * (i + 6) + 2], mp_position[3 * (i + 7) + 2]);

                __m256 pjx = _mm256_set1_ps(mp_position[3 * j]);
                __m256 pjy = _mm256_set1_ps(mp_position[3 * j + 1]);
                __m256 pjz = _mm256_set1_ps(mp_position[3 * j + 2]);

                __m256 rx = _mm256_sub_ps(pjx, pix);
                __m256 ry = _mm256_sub_ps(pjy, piy);
                __m256 rz = _mm256_sub_ps(pjz, piz);

                __m256 eps2 = _mm256_set1_ps(mp_properties->EPS2);

                __m256 rx2 = _mm256_mul_ps(rx, rx);
                __m256 ry2 = _mm256_mul_ps(ry, ry);
                __m256 rz2 = _mm256_mul_ps(rz, rz);
                __m256 rabs = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(rx2, ry2), _mm256_add_ps(rz2, eps2)));

                __m256 m = _mm256_set1_ps(mp_mass[j]);
                __m256 rabsInv = _mm256_div_ps(m, _mm256_mul_ps(_mm256_mul_ps(rabs, rabs), rabs));

                __m256 aix = _mm256_mul_ps(rx, rabsInv);
                __m256 aiy = _mm256_mul_ps(ry, rabsInv);
                __m256 aiz = _mm256_mul_ps(rz, rabsInv);

                for (int k = 0; k < 8; k++) {
                    mp_acceleration[(3 * (i + 7 - k))] = aix.m256_f32[k];
                    mp_acceleration[(3 * (i + 7 - k)) + 1] = aiy.m256_f32[k];
                    mp_acceleration[(3 * (i + 7 - k)) + 2] = aiz.m256_f32[k];
                }
            }
        }*/

        //////////////////////////////////////////////////
        float stepTime2 = 0.5f * mp_properties->stepTime * mp_properties->stepTime;

/*#ifdef OPENMP
#pragma omp parallel for
#endif*/
        for (int i = 0; i < mp_properties->numBody; i++) {
            //3*4 FLOPS
            mp_position[3 * i] += mp_velocity[3 * i] * mp_properties->stepTime + mp_acceleration[3 * i] * stepTime2;
            mp_position[3 * i + 1] += mp_velocity[3 * i + 1] * mp_properties->stepTime + mp_acceleration[3 * i + 1] * stepTime2;
            mp_position[3 * i + 2] += mp_velocity[3 * i + 2] * mp_properties->stepTime + mp_acceleration[3 * i + 2] * stepTime2;
            //3*2 FLOPS
            mp_velocity[3 * i] = mp_velocity[3 * i] * mp_properties->VELOCITY_DAMPENING + mp_acceleration[3 * i] * mp_properties->stepTime;
            mp_velocity[3 * i + 1] = mp_velocity[3 * i + 1] * mp_properties->VELOCITY_DAMPENING + mp_acceleration[3 * i + 1] * mp_properties->stepTime;
            mp_velocity[3 * i + 2] = mp_velocity[3 * i + 2] * mp_properties->VELOCITY_DAMPENING + mp_acceleration[3 * i + 2] * mp_properties->stepTime;
        }
        // numBody * numBody * 23 FLOPS   +   numBody * 18 FLOPS
        mp_properties->currentTime += mp_properties->stepTime;
    }
}