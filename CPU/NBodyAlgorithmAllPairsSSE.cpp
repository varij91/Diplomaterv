#include "NBodyAlgorithmAllPairsSSE.h"
#include "NBodyProperties.h"
#include "omp.h"

void NBodyAlgorithmAllPairsSSE::advance(std::vector<Body> &bodies) {
    /* Új gyorsulási értékek kiszámítása */
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

        for (int j = 0; j < mp_properties->numBody; j++) {
            float3 accI[4] = { zeros, zeros, zeros, zeros };
            calculateAcceleration(posI, bodies.at(j).mass, bodies.at(j).position, accI);
            /*accI[i] = calculateAcceleration(posI[i], bodies.at(j).mass, bodies.at(j).position);
            accI[i+1] = calculateAcceleration(posI[i+1], bodies.at(j).mass, bodies.at(j).position);
            accI[i+2] = calculateAcceleration(posI[i+2], bodies.at(j).mass, bodies.at(j).position);
            accI[i+3] = calculateAcceleration(posI[i+3], bodies.at(j).mass, bodies.at(j).position);*/
            bodies.at(i).acceleration += accI[0];
            bodies.at(i+1).acceleration += accI[1];
            bodies.at(i+2).acceleration += accI[2];
            bodies.at(i+3).acceleration += accI[3];
        }
    }

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
        bodies.at(i).velocity.x += bodies.at(i).acceleration.x * mp_properties->stepTime;
        bodies.at(i).velocity.y += bodies.at(i).acceleration.y * mp_properties->stepTime;
        bodies.at(i).velocity.z += bodies.at(i).acceleration.z * mp_properties->stepTime;
    }

    // numBody * numBody * 23 FLOPS   +   numBody * 18 FLOPS
}