#include "NBodyAlgorithmAllPairs.h"

void NBodyAlgorithmAllPairs::advance(const unsigned int numBody, const float *mass,
    float *pos, float *vel, float *acc, const float stepTime) {
    /* Új gyorsulási értékek kiszámítása */
    for (int i = 0; i < numBody; i++) {

        float posI[3] = { pos[3 * i], pos[3 * i + 1], pos[3 * i + 2] };
        float accI[3] = { 0, 0, 0 };
        acc[3 * i] = 0;
        acc[3 * i + 1] = 0;
        acc[3 * i + 2] = 0;

        for (int j = 0; j < numBody; j++) {
            float posJ[3] = { pos[3 * j], pos[3 * j + 1], pos[3 * j + 2] };
            float massJ   = mass[j];
            calculateAcceleration(posI, accI, massJ, posJ);
            acc[3 * i] += accI[0];
            acc[3 * i + 1] += accI[1];
            acc[3 * i + 2] += accI[2];
        }
    }
    /* Új pozíció és sebesség meghatározása*/
    float stepTime2 = 0.5 * stepTime * stepTime;
    for (int i = 0; i < numBody; i++) {
        pos[3 * i] += vel[3 * i] * stepTime + acc[3 * i] * stepTime2;
        pos[3 * i + 1] += vel[3 * i + 1] * stepTime + acc[3 * i + 1] * stepTime2;
        pos[3 * i + 2] += vel[3 * i + 2] * stepTime + acc[3 * i + 2] * stepTime2;

        vel[3 * i] += acc[3 * i] * stepTime;
        vel[3 * i + 1] += acc[3 * i + 1] * stepTime;
        vel[3 * i + 2] += acc[3 * i + 2] * stepTime;
    }
}