#include "NBodyAlgorithmAllPairsSSE.h"
#include "NBodyProperties.h"
#include "omp.h"

void NBodyAlgorithmAllPairsSSE::advance(std::vector<Body> &bodies) {
    /* Új gyorsulási értékek kiszámítása */
    //float *accI = (float*)_aligned_malloc(12 * sizeof(float), 16);
#ifdef OPENMP
#pragma omp parallel for num_threads(8)
#endif
    for (int i = 0; i < mp_properties->numBody; i+=4) {
       /* if ((i + 4) > mp_properties->numBody) {
            // Ha nem 4-gyel osztható a testek száma az utolsó párat mégegyszer kiszámoljuk
            i = mp_properties->numBody - 4;
        }*/

        float3 zeros = float3(0.0f, 0.0f, 0.0f);

        float3 posI[4] = { bodies.at(i).position, bodies.at(i+1).position,
                           bodies.at(i+2).position, bodies.at(i+3).position };
        

        bodies.at(i).acceleration = zeros;
        bodies.at(i + 1).acceleration = zeros;
        bodies.at(i + 2).acceleration = zeros;
        bodies.at(i + 3).acceleration = zeros;

        
        //float accI[12] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        for (int j = 0; j < mp_properties->numBody; j++) {
            //float accI[12] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
            /*for (int k = 0; k < 12; k++) {
                accI[k] = 0.0f;
            }*/
            /*__m128 currAccx = _mm_set_ps(bodies.at(i).acceleration.x, bodies.at(i + 1).acceleration.x, bodies.at(i + 2).acceleration.x, bodies.at(i + 3).acceleration.x);
            __m128 currAccy = _mm_set_ps(bodies.at(i).acceleration.y, bodies.at(i + 1).acceleration.y, bodies.at(i + 2).acceleration.y, bodies.at(i + 3).acceleration.y);
            __m128 currAccz = _mm_set_ps(bodies.at(i).acceleration.z, bodies.at(i + 1).acceleration.z, bodies.at(i + 2).acceleration.z, bodies.at(i + 3).acceleration.z);*/
            float3 accI[4] = { zeros, zeros, zeros, zeros };
            calculateAcceleration(posI, bodies.at(j).mass, bodies.at(j).position, accI);

            /* TODO IDEIGLENESEN COLOR*/
            /*unsigned int isClose[4] = { 1, 1, 1, 1 };
            calculateAccelerationWithColor(posI, bodies.at(j).mass, bodies.at(j).position, accI, isClose);
            mp_properties->numNeighbours.at(i) += isClose[0];
            mp_properties->numNeighbours.at(i + 1) += isClose[1];
            mp_properties->numNeighbours.at(i + 2) += isClose[2];
            mp_properties->numNeighbours.at(i + 3) += isClose[3];*/

            //calculateAcceleration(posI, bodies.at(j).mass, bodies.at(j).position, currAccx, currAccy, currAccz, accI);
            /*accI[i] = calculateAcceleration(posI[i], bodies.at(j).mass, bodies.at(j).position);
            accI[i+1] = calculateAcceleration(posI[i+1], bodies.at(j).mass, bodies.at(j).position);
            accI[i+2] = calculateAcceleration(posI[i+2], bodies.at(j).mass, bodies.at(j).position);
            accI[i+3] = calculateAcceleration(posI[i+3], bodies.at(j).mass, bodies.at(j).position);*/
            bodies.at(i).acceleration += accI[0];
            bodies.at(i+1).acceleration += accI[1];
            bodies.at(i+2).acceleration += accI[2];
            bodies.at(i+3).acceleration += accI[3];
            /*bodies.at(i).acceleration.x += accI[0];
            bodies.at(i + 1).acceleration.x += accI[1];
            bodies.at(i + 2).acceleration.x += accI[2];
            bodies.at(i + 3).acceleration.x += accI[3];
            bodies.at(i).acceleration.y += accI[4];
            bodies.at(i + 1).acceleration.y += accI[5];
            bodies.at(i + 2).acceleration.y += accI[6];
            bodies.at(i + 3).acceleration.y += accI[7];
            bodies.at(i).acceleration.z += accI[8];
            bodies.at(i + 1).acceleration.z += accI[9];
            bodies.at(i + 2).acceleration.z += accI[10];
            bodies.at(i + 3).acceleration.z += accI[11];*/
            /*bodies.at(i).acceleration.x = accI[0];
            bodies.at(i + 1).acceleration.x = accI[1];
            bodies.at(i + 2).acceleration.x = accI[2];
            bodies.at(i + 3).acceleration.x = accI[3];
            bodies.at(i).acceleration.y = accI[4];
            bodies.at(i + 1).acceleration.y = accI[5];
            bodies.at(i + 2).acceleration.y = accI[6];
            bodies.at(i + 3).acceleration.y = accI[7];
            bodies.at(i).acceleration.z = accI[8];
            bodies.at(i + 1).acceleration.z = accI[9];
            bodies.at(i + 2).acceleration.z = accI[10];
            bodies.at(i + 3).acceleration.z = accI[11];*/
        }
    }
    //_aligned_free(accI);
    /* Új pozíció és sebesség meghatározása*/
    float stepTime2 = 0.5f * mp_properties->stepTime * mp_properties->stepTime;
#ifdef OPENMP
#pragma omp parallel for num_threads(8)
#endif
    for (int i = 0; i < mp_properties->numBody; i++) {
        //3*4 FLOPS
        bodies.at(i).position.x += bodies.at(i).velocity.x * mp_properties->stepTime + bodies.at(i).acceleration.x * stepTime2;
        bodies.at(i).position.y += bodies.at(i).velocity.y * mp_properties->stepTime + bodies.at(i).acceleration.y * stepTime2;
        bodies.at(i).position.z += bodies.at(i).velocity.z * mp_properties->stepTime + bodies.at(i).acceleration.z * stepTime2;
        //3*2 FLOPS
        bodies.at(i).velocity.x = bodies.at(i).velocity.x * mp_properties->velocity_dampening + bodies.at(i).acceleration.x * mp_properties->stepTime;
        bodies.at(i).velocity.y = bodies.at(i).velocity.y * mp_properties->velocity_dampening + bodies.at(i).acceleration.y * mp_properties->stepTime;
        bodies.at(i).velocity.z = bodies.at(i).velocity.z * mp_properties->velocity_dampening + bodies.at(i).acceleration.z * mp_properties->stepTime;
    }

    // numBody * numBody * 23 FLOPS   +   numBody * 18 FLOPS
}