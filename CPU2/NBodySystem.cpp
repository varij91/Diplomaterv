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
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

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


void NBodySystem::setBodyNeighbours() {

}