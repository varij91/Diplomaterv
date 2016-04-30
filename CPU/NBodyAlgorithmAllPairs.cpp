#include "NBodyAlgorithmAllPairs.h"
#include "NBodyProperties.h"
#include "omp.h"

void NBodyAlgorithmAllPairs::advance(std::vector<Body> &bodies) {
    /* Új gyorsulási értékek kiszámítása */
#ifdef OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < mp_properties->numBody; i++) {

        float3 zeros = float3(0.0f, 0.0f, 0.0f);
        bodies.at(i).acceleration = zeros;

        float3 acc(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < mp_properties->numBody; j++) {
            // 17 FLOPS
            acc = calculateAcceleration(bodies.at(i).position, bodies.at(j).mass, bodies.at(j).position);
            // 3 FLOPS
            bodies.at(i).acceleration.x += acc.x;
            bodies.at(i).acceleration.y += acc.y;
            bodies.at(i).acceleration.z += acc.z;
        }
    }

    /* Új pozíció és sebesség meghatározása*/
    float stepTime2 = 0.5f * mp_properties->stepTime * mp_properties->stepTime;
#ifdef OPENMP
#pragma omp parallel for
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