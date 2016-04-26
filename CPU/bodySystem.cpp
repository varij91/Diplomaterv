#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <math.h>
#include "bodySystem.h"
#include "NBodyProperties.h"
#include "NBodyAlgorithmAllPairs.h"
#define GLEW_STATIC
#include <GL\glew.h>
#include <GL\freeglut.h>

int scaledvalue(unsigned int scale) {
    int sign = (rand() % 2) ? -1 : 1;
    int value = (float)(rand() % scale);
    return (sign * value);
}

void BodySystem::init() {
    assert(mp_properties->numBody > 1);
    assert(mp_properties->massScale != 0);
    assert(mp_properties->positionScale != 0);
    assert(mp_properties->velocityScale != 0);
    assert(mp_properties->startTime < mp_properties->endTime);
    assert(!m_systemInitialized);

    srand(mp_properties->seed);
    float3 zeros = float3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < mp_properties->numBody;) {

        m_bodies.emplace_back(zeros, zeros, zeros, 0.0f);

        m_bodies.at(i).position.x = scaledvalue(mp_properties->positionScale);
        m_bodies.at(i).position.y = scaledvalue(mp_properties->positionScale);
        m_bodies.at(i).position.z = scaledvalue(mp_properties->positionScale);

        bool occupied = false;
        float disterr = 10e-6f;
        for (int j = 0; j < i; j++) {
            occupied = (abs(m_bodies.at(j).position.x - m_bodies.at(i).position.x) < disterr) &&
                (abs(m_bodies.at(j).position.y - m_bodies.at(i).position.y) < disterr) &&
                (abs(m_bodies.at(j).position.z - m_bodies.at(i).position.z) < disterr);
            if (occupied) break;
        }
        if (occupied) continue;

        switch (mp_properties->massInit) {
        case(EQUAL) :
            if (i == 0)
                m_bodies.at(i).mass = mp_properties->gravConstant * (float)((rand() % mp_properties->massScale) + 1.0f);
            else
                m_bodies.at(i).mass = m_bodies.at(i - 1).mass;
            break;
        case(RANDOM) :
            m_bodies.at(i).mass = mp_properties->gravConstant * (float)((rand() % mp_properties->massScale) + 1.0f);
            break;
        default :
            assert(false);
            break;
        }
        
        m_bodies.at(i).velocity.x = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);
        m_bodies.at(i).velocity.y = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);
        m_bodies.at(i).velocity.z = mp_properties->initVelocityFactor * scaledvalue(mp_properties->velocityScale);

        m_bodies.at(i).acceleration = zeros;

        i++;
    }

    m_systemInitialized = true;

}

void BodySystem::initGL(int *argc, char* argv[]) {
    glutInit(argc, argv);
    // T�rl�si sz�n be�ll�t�sa
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glutInitWindowSize(650, 650);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    // Pontok sim�t�s�nak be�ll�t�sa
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_FASTEST);
    
    glClearDepth(1.0f);        // Set background depth to farthest
    glEnable(GL_DEPTH_TEST);   // Enable depth testing for z-culling
    glDepthFunc(GL_LEQUAL);    // Set the type of depth-test
    glShadeModel(GL_SMOOTH);   // Enable smooth shading
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);  // Nice perspective corrections

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //gluPerspective(45.0f, aspect, 0.1f, 100.0f);
    /*glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f);*/
    //glOrtho(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);


    glutCreateWindow("N-test szimul�ci�");
}

void BodySystem::setAlgorithm() {
    switch (mp_properties->algorithm) {
    case(ALL_PAIRS) :
        m_algorithm.reset(new NBodyAlgorithmAllPairs(mp_properties));
        m_algorithmInitialized = true;
        break;
    case(ALL_PAIRS_SELECTIVE) :
        break;
    default:
        assert(false);
        break;
    }
}

void BodySystem::integrate() {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);
    
    while (mp_properties->currentTime <= mp_properties->endTime) {
        advance();

        /*std::cout << "#############################################################" << std::endl;
        std::cout << "Time: " << i << std::endl;

        for (int j = 0; j < mp_properties->numBody; j++) {
            std::cout << j << " Pos: (" << m_bodies.at(j).position.x << ", " << m_bodies.at(j).position.y << ", " << m_bodies.at(j).position.z << ")" << std::endl;
        }*/
        /*if (mp_properties->displayMode == DisplayMode::GUI) {
            renderSystem();
        }*/

        /*if ((i + mp_properties->stepTime) > mp_properties->endTime) {
            stepTime = mp_properties->endTime - i;
            i = mp_properties->endTime;
        }
        else {
            i += stepTime;
        }*/
    }

}

void BodySystem::advance() {
    assert(m_systemInitialized);
    assert(m_algorithmInitialized);

    m_algorithm->advance(m_bodies);

    mp_properties->currentTime += mp_properties->stepTime;
}

void BodySystem::renderSystem(void) {

    GLfloat pointSize = 3.0f;
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    
    for (int i = 0; i < mp_properties->numBody; i++)
    {
        glVertex3f(m_bodies.at(i).position.x / (mp_properties->positionScale * 10), m_bodies.at(i).position.y / (mp_properties->positionScale * 10), m_bodies.at(i).position.z / (mp_properties->positionScale * 10));
    }
    glEnd();

    glutSwapBuffers();
    /*// Ablak t�rl�se a kor�bban be�ll�tott sz�nre
    // T�rli a bitmaszknak megfelel� buffer(eke)t
    glClear(GL_COLOR_BUFFER_BIT);
    glPointSize(7.f);
    glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_LOWER_LEFT);
    //glPointParameteri(GL_POINT_SPRITE_COORD_ORIGIN, GL_LOWER_LEFT);
    glEnable(GL_POINT_SPRITE);
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < mp_properties->numBody; i++)
    {
        glVertex3f(bodies.at(i).position.x / (mp_properties->positionScale * 10), bodies.at(i).position.y / (mp_properties->positionScale * 10), bodies.at(i).position.z / (mp_properties->positionScale * 10));
    }
    // Done drawing points
    glEnd();
    
    glutSwapBuffers();*/
}
