#include "NBodyAlgorithmGPU.cuh"

// szop�shegy van az eps2 �tad�s�val (nem akarom argumentumk�nt �tadni minden egyes alkalommal
#define EPS2 10.0f

__device__ float3 tileCalculateAcceleration(const float3 posI) {
    float3 accSumI;
    extern __shared__ float3 posJ[];
    extern __shared__ float massJ[];
    for (int i = 0; i < blockDim.x; i++) {
        accSumI = calculateAcceleration(posI, massJ[i], posJ[i], accSumI);
    }
    return accSumI;
}

__device__ float3 calculateAcceleration(const float3 posI, const float massJ, const float3 posJ, float3 accSumI) {
    float3 r;

    r.x = posJ.x - posI.x;
    r.y = posJ.y - posI.y;
    r.z = posJ.z - posI.z;

    float rabs = sqrt(r.x * r.x + r.y * r.y + r.z * r.z + EPS2);
    float rabsInv = 1.0f / (rabs * rabs * rabs);
    float temp = massJ * rabsInv;

    // A t�megbe bele van olvasztva a G
    // Az rabsInv-be beleraktum a massJ-t, hogy ne kelljen mindig �sszeszorozni
    // Be kelle hozni az gyorul�s �rt�kek akkumul�l�s�t, mert float3/float4-gyel val� m�veleteket nem t�mogatja a CUDA C
    // Szint�n nem elhanyagolhat� hogy MAC m�veletet ki kell haszn�lni, nem aj�nlott az akkumul�l�st k�l�n elv�gezni
    accSumI.x += r.x * temp;
    accSumI.y += r.y * temp;
    accSumI.z += r.z * temp;
    return accSumI;
}
