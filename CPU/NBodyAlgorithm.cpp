#include "NBodyAlgorithm.h"
#include "defines.h"
#include <math.h>

void NBodyAlgorithm::calculateAcceleration(const float posI[3], float accI[3], const float massJ, const float posJ[3]) {
    float r[3];

    r[0] = posJ[0] - posI[0];
    r[1] = posJ[1] - posI[1];
    r[2] = posJ[2] - posI[2];

    float rabs = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + EPS2);
    float rabsInv = 1.0 / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A tömegbe bele van olvaszva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig összeszorozni
    accI[0] += r[0] * temp;
    accI[1] += r[1] * temp;
    accI[2] += r[2] * temp;
}
