#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <math.h>
#include "NBodySystem.h"
#include "NBodyProperties.h"
#define GLEW_STATIC
#include <GL\glew.h>
#include <GL\freeglut.h>

void NBodySystem::init() {
    assert(mp_properties->numBody > 1);
    assert(mp_properties->massScale != 0);
    assert(mp_properties->positionScale != 0);
    assert(mp_properties->velocityScale != 0);
    assert(mp_properties->startTime < mp_properties->endTime);
    //assert(!m_systemInitialized);

    srand(mp_properties->seed);
    float3 zeros;
    zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f;

    float disterr = 10e-6f;

    for (int i = 0; i < mp_properties->numBody;) {

        m_bodies.emplace_back(zeros, zeros, zeros, 0.0f);

        m_bodies.at(i).position = mp_initializator->getNewPosition();

        // Ellenõrizzük, hogy ne kerüljön egymáshoz túl közel két test
        // Bizonyos beállítások alkalmazása mellett, elõfordulhat, hogy kifagy a rendszer
        bool occupied = false;
        for (int j = 0; j < i; j++) {
            occupied = (abs(m_bodies.at(j).position.x - m_bodies.at(i).position.x) < disterr) &&
                (abs(m_bodies.at(j).position.y - m_bodies.at(i).position.y) < disterr) &&
                (abs(m_bodies.at(j).position.z - m_bodies.at(i).position.z) < disterr);
            if (occupied) break;
        }
        if (occupied) continue;

        m_bodies.at(i).mass = mp_initializator->getNewMass();
        
        m_bodies.at(i).velocity = mp_initializator->getNewVelocity();

        m_bodies.at(i).acceleration = zeros;

        i++;
    }

    m_systemInitialized = true;

}

void NBodySystem::setAlgorithm() {
    mp_initializator->getNewAlgorithm(mp_algorithm);
    m_algorithmInitialized = true;
}

void NBodySystem::integrate() {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);
    while (mp_properties->currentTime < mp_properties->endTime) {
        advance();
    }
}

void NBodySystem::advance() {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);

    // Új pozíció, sebesség, gyorsulásparaméterek meghatározása
    mp_algorithm->advance(m_bodies);

    // Szimuláció továbbléptetése
    mp_properties->currentTime += mp_properties->stepTime;
}

#ifdef USE_FLAT_FUNC
// Próba integrate függvény a függvényhívások okozta esetleges overhead felderítésére
// All_pairs, AVX, Performance, OpenMP
void NBodySystem::integrateFlat() {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);

    float stepTime2 = 0.5f * mp_properties->stepTime * mp_properties->stepTime;

    while (mp_properties->currentTime < mp_properties->endTime) {
        /////////////////////////////////////////////
        if (mp_properties->technology == Technology::BASIC) {

            if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for
#endif
                for (int i = 0; i < mp_properties->numBody; i++) {
                    integrateFlatBasicCore(i);
                }
            }
            else {
                for (int i = 0; i < mp_properties->numBody; i++) {
                    integrateFlatBasicCore(i);
                }
            }
        }
        /////////////////////////////////////////////
        else if (mp_properties->technology == Technology::SSE) {
            if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for
#endif
                for (int i = 0; i < mp_properties->numBody; i += 4) {
                    integrateFlatSSECore(i);
                    /*float3 zeros;
                    zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f;

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

                        for (int k = 0; k < 4; k++) {
                            accI[3 - k].x = aix.m128_f32[k];
                            accI[3 - k].y = aiy.m128_f32[k];
                            accI[3 - k].z = aiz.m128_f32[k];
                        }

                        for (int k = 0; k < 4; k++) {
                            m_bodies.at(i + k).acceleration.x += accI[k].x;
                            m_bodies.at(i + k).acceleration.y += accI[k].y;
                            m_bodies.at(i + k).acceleration.z += accI[k].z;
                        }
                    }*/
                }
            }
            else {
                for (int i = 0; i < mp_properties->numBody; i += 4) {
                    integrateFlatSSECore(i);
                }
            }
        }
        /////////////////////////////////////////////
        else if (mp_properties->technology == Technology::AVX) {
            if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for
#endif
                for (int i = 0; i < mp_properties->numBody; i += 8) {
                    integrateFlatAVXCore(i);
                }
            }
            else {

                for (int i = 0; i < mp_properties->numBody; i += 8) {
                    integrateFlatAVXCore(i);
                }
            }
        }

        //////////////////////////////////////////////////
        
        for (int i = 0; i < mp_properties->numBody; i++) {
                //3*4 FLOPS
            m_bodies.at(i).position.x += m_bodies.at(i).velocity.x * mp_properties->stepTime + m_bodies.at(i).acceleration.x * stepTime2;
            m_bodies.at(i).position.y += m_bodies.at(i).velocity.y * mp_properties->stepTime + m_bodies.at(i).acceleration.y * stepTime2;
            m_bodies.at(i).position.z += m_bodies.at(i).velocity.z * mp_properties->stepTime + m_bodies.at(i).acceleration.z * stepTime2;
                //3*2 FLOPS
            m_bodies.at(i).velocity.x = m_bodies.at(i).velocity.x * mp_properties->VELOCITY_DAMPENING + m_bodies.at(i).acceleration.x * mp_properties->stepTime;
            m_bodies.at(i).velocity.y = m_bodies.at(i).velocity.y * mp_properties->VELOCITY_DAMPENING + m_bodies.at(i).acceleration.y * mp_properties->stepTime;
            m_bodies.at(i).velocity.z = m_bodies.at(i).velocity.z * mp_properties->VELOCITY_DAMPENING + m_bodies.at(i).acceleration.z * mp_properties->stepTime;
        }
            // numBody * numBody * 23 FLOPS   +   numBody * 18 FLOPS
        mp_properties->currentTime += mp_properties->stepTime;
    }
}

inline void NBodySystem::integrateFlatBasicCore(int index) {
    float3 zeros;
    zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f;
    m_bodies.at(index).acceleration = zeros;

    float3 acc(zeros);
    for (int j = 0; j < mp_properties->numBody; j++) {
        // 17 FLOPS
        float3 r(zeros);

        r.x = m_bodies.at(j).position.x - m_bodies.at(index).position.x;
        r.y = m_bodies.at(j).position.y - m_bodies.at(index).position.y;
        r.z = m_bodies.at(j).position.z - m_bodies.at(index).position.z;

        float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + mp_properties->EPS2);
        float rabsInv = 1.0f / (rabs * rabs * rabs);
        float temp = m_bodies.at(j).mass * rabsInv;

        // A tömegbe bele van olvasztva a G
        // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
        acc.x = r.x * temp;
        acc.y = r.y * temp;
        acc.z = r.z * temp;

        // 3 FLOPS
        m_bodies.at(index).acceleration.x += acc.x;
        m_bodies.at(index).acceleration.y += acc.y;
        m_bodies.at(index).acceleration.z += acc.z;
    }
}

inline void NBodySystem::integrateFlatSSECore(int index) {
    float3 zeros;
    zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f;

    float3 posI[4] = { m_bodies.at(index).position, m_bodies.at(index + 1).position,
        m_bodies.at(index + 2).position, m_bodies.at(index + 3).position };

    m_bodies.at(index).acceleration = zeros;
    m_bodies.at(index + 1).acceleration = zeros;
    m_bodies.at(index + 2).acceleration = zeros;
    m_bodies.at(index + 3).acceleration = zeros;

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

        for (int k = 0; k < 4; k++) {
            accI[3 - k].x = aix.m128_f32[k];
            accI[3 - k].y = aiy.m128_f32[k];
            accI[3 - k].z = aiz.m128_f32[k];
        }

        for (int k = 0; k < 4; k++) {
            m_bodies.at(index + k).acceleration.x += accI[k].x;
            m_bodies.at(index + k).acceleration.y += accI[k].y;
            m_bodies.at(index + k).acceleration.z += accI[k].z;
        }
    }
}

inline void NBodySystem::integrateFlatAVXCore(int index) {
    float3 zeros;
    zeros.x = 0.0f; zeros.y = 0.0f; zeros.z = 0.0f;

    float3 posI[8] = { m_bodies.at(index).position, m_bodies.at(index + 1).position,
        m_bodies.at(index + 2).position, m_bodies.at(index + 3).position,
        m_bodies.at(index + 4).position, m_bodies.at(index + 5).position,
        m_bodies.at(index + 6).position, m_bodies.at(index + 7).position };

    m_bodies.at(index).acceleration = zeros;
    m_bodies.at(index + 1).acceleration = zeros;
    m_bodies.at(index + 2).acceleration = zeros;
    m_bodies.at(index + 3).acceleration = zeros;
    m_bodies.at(index + 4).acceleration = zeros;
    m_bodies.at(index + 5).acceleration = zeros;
    m_bodies.at(index + 6).acceleration = zeros;
    m_bodies.at(index + 7).acceleration = zeros;

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

        for (int k = 0; k < 8; k++) {
            accI[7 - k].x = aix.m256_f32[k];
            accI[7 - k].y = aiy.m256_f32[k];
            accI[7 - k].z = aiz.m256_f32[k];
        }

        for (int k = 0; k < 8; k++) {
            m_bodies.at(index + k).acceleration.x += accI[k].x;
            m_bodies.at(index + k).acceleration.y += accI[k].y;
            m_bodies.at(index + k).acceleration.z += accI[k].z;
        }
    }
}
#endif