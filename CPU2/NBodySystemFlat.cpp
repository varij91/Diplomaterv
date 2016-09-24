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
    float3 pos(0.0f, 0.0f, 0.0f);
    float3 vel(0.0f, 0.0f, 0.0f);

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
    assert(m_algorithmInitialized);
    while (mp_properties->currentTime < mp_properties->endTime) {

#ifdef OPENMP
#pragma omp parallel for
#endif
        /////////////////////////////////////////////

        for (int i = 0; i < mp_properties->numBody; i++) {
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
        }
        /////////////////////////////////////////////
        /*for (int i = 0; i < mp_properties->numBody; i += 4) {
        float3 zeros = float3(0.0f, 0.0f, 0.0f);

        float3 posI[4] = { m_bodies.at(i).position, m_bodies.at(i + 1).position,
        m_bodies.at(i + 2).position, m_bodies.at(i + 3).position };

        m_bodies.at(i).acceleration = zeros;
        m_bodies.at(i + 1).acceleration = zeros;
        m_bodies.at(i + 2).acceleration = zeros;
        m_bodies.at(i + 3).acceleration = zeros;

        for (int j = 0; j < mp_properties->numBody; j++) {

        float3 accI[4] = { zeros, zeros, zeros, zeros };

        __m128 pix = _mm_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x);
        __m128 piy = _mm_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y);
        __m128 piz = _mm_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z);

        __m128 pjx = _mm_set_ps1(m_bodies.at(j).position.x);
        __m128 pjy = _mm_set_ps1(m_bodies.at(j).position.y);
        __m128 pjz = _mm_set_ps1(m_bodies.at(j).position.z);

        __m128 rx = _mm_sub_ps(pjx, pix);
        __m128 ry = _mm_sub_ps(pjy, piy);
        __m128 rz = _mm_sub_ps(pjz, piz);

        __m128 eps2 = _mm_set_ps1(mp_properties->EPS2);

        __m128 rx2 = _mm_mul_ps(rx, rx);
        __m128 ry2 = _mm_mul_ps(ry, ry);
        __m128 rz2 = _mm_mul_ps(rz, rz);
        __m128 rabs = _mm_sqrt_ps(_mm_add_ps(_mm_add_ps(rx2, ry2), _mm_add_ps(rz2, eps2)));

        __m128 m = _mm_set_ps1(m_bodies.at(j).mass);
        __m128 rabsInv = _mm_div_ps(m, _mm_mul_ps(_mm_mul_ps(rabs, rabs), rabs));

        __m128 aix = _mm_mul_ps(rx, rabsInv);
        __m128 aiy = _mm_mul_ps(ry, rabsInv);
        __m128 aiz = _mm_mul_ps(rz, rabsInv);

        for (int i = 0; i < 4; i++) {
        accI[3 - i].x = aix.m128_f32[i];
        accI[3 - i].y = aiy.m128_f32[i];
        accI[3 - i].z = aiz.m128_f32[i];
        }

        m_bodies.at(i).acceleration += accI[0];
        m_bodies.at(i + 1).acceleration += accI[1];
        m_bodies.at(i + 2).acceleration += accI[2];
        m_bodies.at(i + 3).acceleration += accI[3];
        }
        }*/
        /////////////////////////////////////////////
        /*for (int i = 0; i < mp_properties->numBody; i += 8) {
            float3 zeros = float3(0.0f, 0.0f, 0.0f);

            float3 posI[8] = { m_bodies.at(i).position, m_bodies.at(i + 1).position,
                m_bodies.at(i + 2).position, m_bodies.at(i + 3).position,
                m_bodies.at(i + 4).position, m_bodies.at(i + 5).position,
                m_bodies.at(i + 6).position, m_bodies.at(i + 7).position };

            m_bodies.at(i).acceleration = zeros;
            m_bodies.at(i + 1).acceleration = zeros;
            m_bodies.at(i + 2).acceleration = zeros;
            m_bodies.at(i + 3).acceleration = zeros;
            m_bodies.at(i + 4).acceleration = zeros;
            m_bodies.at(i + 5).acceleration = zeros;
            m_bodies.at(i + 6).acceleration = zeros;
            m_bodies.at(i + 7).acceleration = zeros;

            for (int j = 0; j < mp_properties->numBody; j++) {

                float3 accI[8] = { zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros };

                __m256 pix = _mm256_set_ps(posI[0].x, posI[1].x, posI[2].x, posI[3].x, posI[4].x, posI[5].x, posI[6].x, posI[7].x);
                __m256 piy = _mm256_set_ps(posI[0].y, posI[1].y, posI[2].y, posI[3].y, posI[4].y, posI[5].y, posI[6].y, posI[7].y);
                __m256 piz = _mm256_set_ps(posI[0].z, posI[1].z, posI[2].z, posI[3].z, posI[4].z, posI[5].z, posI[6].z, posI[7].z);

                __m256 pjx = _mm256_set1_ps(m_bodies.at(j).position.x);
                __m256 pjy = _mm256_set1_ps(m_bodies.at(j).position.y);
                __m256 pjz = _mm256_set1_ps(m_bodies.at(j).position.z);

                __m256 rx = _mm256_sub_ps(pjx, pix);
                __m256 ry = _mm256_sub_ps(pjy, piy);
                __m256 rz = _mm256_sub_ps(pjz, piz);

                __m256 eps2 = _mm256_set1_ps(mp_properties->EPS2);

                __m256 rx2 = _mm256_mul_ps(rx, rx);
                __m256 ry2 = _mm256_mul_ps(ry, ry);
                __m256 rz2 = _mm256_mul_ps(rz, rz);
                __m256 rabs = _mm256_sqrt_ps(_mm256_add_ps(_mm256_add_ps(rx2, ry2), _mm256_add_ps(rz2, eps2)));

                __m256 m = _mm256_set1_ps(m_bodies.at(j).mass);
                __m256 rabsInv = _mm256_div_ps(m, _mm256_mul_ps(_mm256_mul_ps(rabs, rabs), rabs));

                __m256 aix = _mm256_mul_ps(rx, rabsInv);
                __m256 aiy = _mm256_mul_ps(ry, rabsInv);
                __m256 aiz = _mm256_mul_ps(rz, rabsInv);

                for (int i = 0; i < 8; i++) {
                    accI[7 - i].x = aix.m256_f32[i];
                    accI[7 - i].y = aiy.m256_f32[i];
                    accI[7 - i].z = aiz.m256_f32[i];
                }

                m_bodies.at(i).acceleration += accI[0];
                m_bodies.at(i + 1).acceleration += accI[1];
                m_bodies.at(i + 2).acceleration += accI[2];
                m_bodies.at(i + 3).acceleration += accI[3];
                m_bodies.at(i + 4).acceleration += accI[4];
                m_bodies.at(i + 5).acceleration += accI[5];
                m_bodies.at(i + 6).acceleration += accI[6];
                m_bodies.at(i + 7).acceleration += accI[7];
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