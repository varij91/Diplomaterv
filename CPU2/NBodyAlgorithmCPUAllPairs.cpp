#include "NBodyAlgorithmCPUAllPairs.h"

inline void NBodyAlgorithmCPUAllPairs::updateBodies(std::vector<Body> &bodies, int index, int stepTime2) {
    //3*4 FLOPS
    bodies.at(index).position.x += bodies.at(index).velocity.x * mp_properties->stepTime + bodies.at(index).acceleration.x * stepTime2;
    bodies.at(index).position.y += bodies.at(index).velocity.y * mp_properties->stepTime + bodies.at(index).acceleration.y * stepTime2;
    bodies.at(index).position.z += bodies.at(index).velocity.z * mp_properties->stepTime + bodies.at(index).acceleration.z * stepTime2;
    //3*2 FLOPS
    bodies.at(index).velocity.x = bodies.at(index).velocity.x * mp_properties->VELOCITY_DAMPENING + bodies.at(index).acceleration.x * mp_properties->stepTime;
    bodies.at(index).velocity.y = bodies.at(index).velocity.y * mp_properties->VELOCITY_DAMPENING + bodies.at(index).acceleration.y * mp_properties->stepTime;
    bodies.at(index).velocity.z = bodies.at(index).velocity.z * mp_properties->VELOCITY_DAMPENING + bodies.at(index).acceleration.z * mp_properties->stepTime;
}
void NBodyAlgorithmCPUAllPairs::advance(std::vector<Body> &bodies) {
    if (mp_properties->technology == BASIC) {
        advanceBasic(bodies);
    }
    else if (mp_properties->technology == SSE) {
        advanceSSE(bodies);
    }
    else if (mp_properties->technology == AVX) {
        advanceAVX(bodies);
    }

    //////////////////////////////////////////////////
    float stepTime2 = 0.5f * mp_properties->stepTime * mp_properties->stepTime;

    if (mp_properties->useOpenMP) {  
#ifdef OPENMP
#pragma omp parallel for
#endif
        for (int i = 0; i < mp_properties->numBody; i++) {
            updateBodies(bodies, i, stepTime2);
        }
        // numBody * numBody * 23 FLOPS   +   numBody * 18 FLOPS
    }
    else {
        for (int i = 0; i < mp_properties->numBody; i++) {
            updateBodies(bodies, i, stepTime2);
        }
        // numBody * numBody * 23 FLOPS   +   numBody * 18 FLOPS
    }
}

inline void NBodyAlgorithmCPUAllPairs::advanceBasicCore(std::vector<Body> &bodies, int index) {
    float3 zeros = float3(0.0f, 0.0f, 0.0f);
    bodies.at(index).acceleration = zeros;

    float3 acc(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < mp_properties->numBody; j++) {
        // 17 FLOPS
        acc = calculateAcceleration(bodies.at(index).position, bodies.at(j).mass, bodies.at(j).position);
        // 3 FLOPS
        bodies.at(index).acceleration.x += acc.x;
        bodies.at(index).acceleration.y += acc.y;
        bodies.at(index).acceleration.z += acc.z;
    }
}
inline void NBodyAlgorithmCPUAllPairs::advanceBasicCoreGUI(std::vector<Body> &bodies, int index) {
    float3 zeros = float3(0.0f, 0.0f, 0.0f);
    bodies.at(index).acceleration = zeros;

    float3 acc(0.0f, 0.0f, 0.0f);
    unsigned int numNeighbours = 0;
    for (int j = 0; j < mp_properties->numBody; j++) {
        // 17 FLOPS
        acc = calculateAccelerationWithColor(bodies.at(index).position, bodies.at(j).mass, bodies.at(j).position, numNeighbours);
        // 3 FLOPS
        bodies.at(index).acceleration.x += acc.x;
        bodies.at(index).acceleration.y += acc.y;
        bodies.at(index).acceleration.z += acc.z;
    }
    mp_properties->numNeighbours.at(index) = numNeighbours;
}
void NBodyAlgorithmCPUAllPairs::advanceBasic(std::vector<Body> &bodies) {
    /* Új gyorsulási értékek kiszámítása */
    if (mp_properties->mode == GUI) {
        if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < mp_properties->numBody; i++) {
                advanceBasicCoreGUI(bodies, i);
            }
        }
        else {
            for (int i = 0; i < mp_properties->numBody; i++) {
                advanceBasicCoreGUI(bodies, i);
            }
        }
    }
    else {
        if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for
#endif
            for (int i = 0; i < mp_properties->numBody; i++) {
                advanceBasicCore(bodies, i);
            }
        }
        else {
            for (int i = 0; i < mp_properties->numBody; i++) {
                advanceBasicCore(bodies, i);
            }
        }
    }


}

inline void NBodyAlgorithmCPUAllPairs::advanceSSECore(std::vector<Body> &bodies, int index) {
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

    float3 posI[4] = { bodies.at(index).position, bodies.at(index + 1).position,
        bodies.at(index + 2).position, bodies.at(index + 3).position };

    bodies.at(index).acceleration = zeros;
    bodies.at(index + 1).acceleration = zeros;
    bodies.at(index + 2).acceleration = zeros;
    bodies.at(index + 3).acceleration = zeros;

    for (int j = 0; j < mp_properties->numBody; j++) {

        float3 accI[4] = { zeros, zeros, zeros, zeros };
        calculateAcceleration(posI, bodies.at(j).mass, bodies.at(j).position, accI);

        bodies.at(index).acceleration += accI[0];
        bodies.at(index + 1).acceleration += accI[1];
        bodies.at(index + 2).acceleration += accI[2];
        bodies.at(index + 3).acceleration += accI[3];
    }
}
inline void NBodyAlgorithmCPUAllPairs::advanceSSECoreGUI(std::vector<Body> &bodies, int index) {
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

    float3 posI[4] = { bodies.at(index).position, bodies.at(index + 1).position,
        bodies.at(index + 2).position, bodies.at(index + 3).position };

    bodies.at(index).acceleration = zeros;
    bodies.at(index + 1).acceleration = zeros;
    bodies.at(index + 2).acceleration = zeros;
    bodies.at(index + 3).acceleration = zeros;

    mp_properties->numNeighbours.at(index) = 0;
    mp_properties->numNeighbours.at(index + 1) = 0;
    mp_properties->numNeighbours.at(index + 2) = 0;
    mp_properties->numNeighbours.at(index + 3) = 0;

    for (int j = 0; j < mp_properties->numBody; j++) {

        unsigned int numNeighbours[4] = { 1, 1, 1, 1 };

        float3 accI[4] = { zeros, zeros, zeros, zeros };
        calculateAccelerationWithColor(posI, bodies.at(j).mass, bodies.at(j).position, accI, numNeighbours);

        bodies.at(index).acceleration += accI[0];
        bodies.at(index + 1).acceleration += accI[1];
        bodies.at(index + 2).acceleration += accI[2];
        bodies.at(index + 3).acceleration += accI[3];

        mp_properties->numNeighbours.at(index) += numNeighbours[0];
        mp_properties->numNeighbours.at(index + 1) += numNeighbours[1];
        mp_properties->numNeighbours.at(index + 2) += numNeighbours[2];
        mp_properties->numNeighbours.at(index + 3) += numNeighbours[3];
    }


}
void NBodyAlgorithmCPUAllPairs::advanceSSE(std::vector<Body> &bodies) {
    /* Új gyorsulási értékek kiszámítása */
    /* if ((i + 4) > mp_properties->numBody) {
    // Ha nem 4-gyel osztható a testek száma az utolsó párat mégegyszer kiszámoljuk
    i = mp_properties->numBody - 4;
    }*/
    if (mp_properties->mode == GUI) {
        if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for num_threads(8)
#endif
            for (int i = 0; i < mp_properties->numBody; i += 4) {
                advanceSSECoreGUI(bodies, i);
            }
        }
        else {

            for (int i = 0; i < mp_properties->numBody; i += 4) {
                advanceSSECoreGUI(bodies, i);
            }
        }
    }
    else {
        if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for num_threads(8)
#endif
            for (int i = 0; i < mp_properties->numBody; i += 4) {
                advanceSSECore(bodies, i);
            }
        }
        else {

            for (int i = 0; i < mp_properties->numBody; i += 4) {
                advanceSSECore(bodies, i);
            }
        }
    }


}

inline void NBodyAlgorithmCPUAllPairs::advanceAVXCore(std::vector<Body> &bodies, int index) {
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

    float3 posI[8] = { bodies.at(index).position, bodies.at(index + 1).position,
        bodies.at(index + 2).position, bodies.at(index + 3).position,
        bodies.at(index + 4).position, bodies.at(index + 5).position,
        bodies.at(index + 6).position, bodies.at(index + 7).position };

    bodies.at(index).acceleration = zeros;
    bodies.at(index + 1).acceleration = zeros;
    bodies.at(index + 2).acceleration = zeros;
    bodies.at(index + 3).acceleration = zeros;
    bodies.at(index + 4).acceleration = zeros;
    bodies.at(index + 5).acceleration = zeros;
    bodies.at(index + 6).acceleration = zeros;
    bodies.at(index + 7).acceleration = zeros;

    for (int j = 0; j < mp_properties->numBody; j++) {

        float3 accI[8] = { zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros };
        calculateAcceleration(posI, bodies.at(j).mass, bodies.at(j).position, accI);

        bodies.at(index).acceleration += accI[0];
        bodies.at(index + 1).acceleration += accI[1];
        bodies.at(index + 2).acceleration += accI[2];
        bodies.at(index + 3).acceleration += accI[3];
        bodies.at(index + 4).acceleration += accI[4];
        bodies.at(index + 5).acceleration += accI[5];
        bodies.at(index + 6).acceleration += accI[6];
        bodies.at(index + 7).acceleration += accI[7];
    }
}
inline void NBodyAlgorithmCPUAllPairs::advanceAVXCoreGUI(std::vector<Body> &bodies, int index) {
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

    float3 posI[8] = { bodies.at(index).position, bodies.at(index + 1).position,
        bodies.at(index + 2).position, bodies.at(index + 3).position,
        bodies.at(index + 4).position, bodies.at(index + 5).position,
        bodies.at(index + 6).position, bodies.at(index + 7).position };

    bodies.at(index).acceleration = zeros;
    bodies.at(index + 1).acceleration = zeros;
    bodies.at(index + 2).acceleration = zeros;
    bodies.at(index + 3).acceleration = zeros;
    bodies.at(index + 4).acceleration = zeros;
    bodies.at(index + 5).acceleration = zeros;
    bodies.at(index + 6).acceleration = zeros;
    bodies.at(index + 7).acceleration = zeros;

    mp_properties->numNeighbours.at(index) = 0;
    mp_properties->numNeighbours.at(index + 1) = 0;
    mp_properties->numNeighbours.at(index + 2) = 0;
    mp_properties->numNeighbours.at(index + 3) = 0;
    mp_properties->numNeighbours.at(index + 4) = 0;
    mp_properties->numNeighbours.at(index + 5) = 0;
    mp_properties->numNeighbours.at(index + 6) = 0;
    mp_properties->numNeighbours.at(index + 7) = 0;

    for (int j = 0; j < mp_properties->numBody; j++) {

        unsigned int numNeighbours[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };

        float3 accI[8] = { zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros };
        calculateAccelerationWithColor(posI, bodies.at(j).mass, bodies.at(j).position, accI, numNeighbours);

        bodies.at(index).acceleration += accI[0];
        bodies.at(index + 1).acceleration += accI[1];
        bodies.at(index + 2).acceleration += accI[2];
        bodies.at(index + 3).acceleration += accI[3];
        bodies.at(index + 4).acceleration += accI[4];
        bodies.at(index + 5).acceleration += accI[5];
        bodies.at(index + 6).acceleration += accI[6];
        bodies.at(index + 7).acceleration += accI[7];

        mp_properties->numNeighbours.at(index) += numNeighbours[0];
        mp_properties->numNeighbours.at(index + 1) += numNeighbours[1];
        mp_properties->numNeighbours.at(index + 2) += numNeighbours[2];
        mp_properties->numNeighbours.at(index + 3) += numNeighbours[3];
        mp_properties->numNeighbours.at(index + 4) += numNeighbours[4];
        mp_properties->numNeighbours.at(index + 5) += numNeighbours[5];
        mp_properties->numNeighbours.at(index + 6) += numNeighbours[6];
        mp_properties->numNeighbours.at(index + 7) += numNeighbours[7];
    }


}
void NBodyAlgorithmCPUAllPairs::advanceAVX(std::vector<Body> &bodies) {
    /* Új gyorsulási értékek kiszámítása */
    /* if ((i + 8) > mp_properties->numBody) {
    // Ha nem 8-gyel osztható a testek száma az utolsó párat mégegyszer kiszámoljuk
    i = mp_properties->numBody - 8;
    }*/
    if (mp_properties->mode == GUI) {
        if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for num_threads(8)
#endif
            for (int i = 0; i < mp_properties->numBody; i += 8) {
                advanceAVXCoreGUI(bodies, i);
            }
        }
        else {

            for (int i = 0; i < mp_properties->numBody; i += 8) {
                advanceAVXCoreGUI(bodies, i);
            }
        }
    }
    else {
        if (mp_properties->useOpenMP) {
#ifdef OPENMP
#pragma omp parallel for num_threads(8)
#endif
            for (int i = 0; i < mp_properties->numBody; i += 8) {
                advanceAVXCore(bodies, i);
            }
        }
        else {

            for (int i = 0; i < mp_properties->numBody; i += 8) {
                advanceAVXCore(bodies, i);
            }
        }
    }
}