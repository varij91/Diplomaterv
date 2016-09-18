#include <stdio.h>
#include <iostream>
#include <assert.h>
#include <random>
#include <math.h>
#include "bodySystem.h"
#include "NBodyProperties.h"
#include "NBodyAlgorithmAllPairs.h"
#include "NBodyAlgorithmAllPairsSSE.h"
#define GLEW_STATIC
#include <GL\glew.h>
#include <GL\freeglut.h>

float scaledvalue(unsigned int scale) {
    float sign = (rand() % 2) ? -1.0f : 1.0f;
    float integer  = (float)(rand() % scale);
    float rmax = (float)RAND_MAX;
    float fraction = (float)(rand() % RAND_MAX) / rmax;
    
    return (sign * (integer + fraction));
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
    // Törlési szín beállítása
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glutInitWindowSize(720, 720);
    glutInitWindowPosition(0, 0);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    // Pontok simításának beállítása
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


    glutCreateWindow("N-test szimuláció");
}

void BodySystem::setAlgorithm() {
    switch (mp_properties->algorithm) {
    case(ALL_PAIRS) :
        m_algorithm.reset(new NBodyAlgorithmAllPairsSSE(mp_properties));
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
    /*for (int j = 0; j < mp_properties->numBody; j++) {
        std::cout << j << " Pos: (" << m_bodies.at(j).position.x << ", " << m_bodies.at(j).position.y << ", " << m_bodies.at(j).position.z << ")" << std::endl;
        std::cout << j << " Acc: (" << m_bodies.at(j).acceleration.x << ", " << m_bodies.at(j).acceleration.y << ", " << m_bodies.at(j).acceleration.z << ")" << std::endl;
    }*/
    while (mp_properties->currentTime < mp_properties->endTime) {
        advance();

        /*std::cout << "#############################################################" << std::endl;
        std::cout << "Time: " << mp_properties->currentTime << std::endl;

        for (int j = 0; j < mp_properties->numBody; j++) {
            std::cout << j << " Pos: (" << m_bodies.at(j).position.x << ", " << m_bodies.at(j).position.y << ", " << m_bodies.at(j).position.z << ")" << std::endl;
            std::cout << j << " Acc: (" << m_bodies.at(j).acceleration.x << ", " << m_bodies.at(j).acceleration.y << ", " << m_bodies.at(j).acceleration.z << ")" << std::endl;
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


void BodySystem::getColor(unsigned int &r, unsigned int &g, unsigned int &b, int index) {
    /*unsigned int value = (mp_properties->numNeighbours.at(index) * 256) / mp_properties->numBody;
    unsigned int quotient = value / 256;
    unsigned int remainder = value % 256;
    if (quotient == 0) {
        r = remainder; g = 0; b = 255;
    }
    else if (quotient == 1) {
        r = 255; g = 0; b = 255-remainder;
    }
    else if(quotient == 2) {
        r = 255; g = remainder; b = 0;
    }
    else {
        r = 255; g = 255; b = 0;
    }*/
    float value = (float)mp_properties->numNeighbours.at(index) / (float)mp_properties->numBody;
    if (mp_properties->numBody <= 128)
        value = value * 0.25f;
    else if (mp_properties->numBody <= 256)
        value = value * 0.5f;
    else if (mp_properties->numBody <= 512)
        value = value * 1.0f;
    else if (mp_properties->numBody <= 1024)
        value = value * 2.2f;
    else if (mp_properties->numBody <= 2048)
        value = value * 2.5f;
    else if (mp_properties->numBody <= 4096)
        value = value * 5.0f;
    else
        value = value * 16.0f;

    if (value < 0.001f) {
        r = 0; g = 0; b = 255;
    }
    else if (value < 0.002f) {
        r = 127; g = 0; b = 255;
    }
    else if (value < 0.004f) {
        r = 255; g = 0; b = 255;
    }
    else if (value < 0.008f) {
        r = 255; g = 0; b = 170;
    }
    else if (value < 0.016f) {
        r = 255; g = 0; b = 85;
    }
    else if (value < 0.032f) {
        r = 255; g = 0; b = 0;
    }
    else if (value < 0.064f) {
        r = 255; g = 31; b = 0;
    }
    else if (value < 0.128f) {
        r = 255; g = 63; b = 0;
    }
    else if (value < 0.256f) {
        r = 255; g = 95; b = 0;
    }
    else if (value < 0.384f) {
        r = 255; g = 127; b = 0;
    }
    else if (value < 0.512f) {
        r = 255; g = 159; b = 0;
    }
    else if (value < 0.640f) {
        r = 255; g = 191; b = 0;
    }
    else if (value < 0.768f) {
        r = 255; g = 223; b = 0;
    }
    else {
        r = 255; g = 255; b = 0;
    }
}

void BodySystem::renderSystem(void) {
    //glClear(GL_COLOR_BUFFER_BIT);
    //glEnable(GL_TEXTURE_2D);
    //glBindTexture(GL_TEXTURE_2D, 1);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    //float size = mp_properties->positionScale/500.0f;

    ///*glEnable(GL_BLEND);
    //glEnable(GL_POLYGON_SMOOTH);
    //glBlendFunc(GL_SRC_ALPHA_SATURATE,GL_ZERO);
    //glHint(GL_POLYGON_SMOOTH_HINT,GL_FASTEST);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glEnable(GL_MULTISAMPLE_ARB);*/

    //glMatrixMode(GL_PROJECTION);
    //glLoadIdentity();
    //glFrustum(-0.1, 0.1, -0.1, 0.1, 0.1, 100.0);
    //glMatrixMode(GL_MODELVIEW);
    //glLoadIdentity();

    //glEnable(GL_ALPHA_TEST);
    //glEnable(GL_POLYGON_SMOOTH);
    //glEnable(GL_POINT_SMOOTH);
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    ////glPointSize(4.0f);
    ///*glBegin(GL_POINTS);

    //for (int i = 0; i < mp_properties->numBody; i++)
    //{
    //    glVertex3f(m_bodies.at(i).position.x, m_bodies.at(i).position.y, m_bodies.at(i).position.z);
    //}
    //glEnd();*/
    //glBegin(GL_QUADS);
    //for (int i = 0; i < mp_properties->numBody; i++)
    //{

    //    glTexCoord2f(0, 0);
    //    glVertex3f(m_bodies.at(i).position.x - size, m_bodies.at(i).position.y - size, m_bodies.at(i).position.z);
    //    glTexCoord2f(0, 1);
    //    glVertex3f(m_bodies.at(i).position.x - size, m_bodies.at(i).position.y + size, m_bodies.at(i).position.z);
    //    glTexCoord2f(1, 1);
    //    glVertex3f(m_bodies.at(i).position.x + size, m_bodies.at(i).position.y + size, m_bodies.at(i).position.z);
    //    glTexCoord2f(1, 0);
    //    glVertex3f(m_bodies.at(i).position.x + size, m_bodies.at(i).position.y - size, m_bodies.at(i).position.z);
    //    
    //}
    //glEnd();
    //glutSwapBuffers();

    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_POINT_SMOOTH);
    //glPointSize(9.0f);
    /*glEnable(GL_FOG);
    glFogf(GL_FOG_MODE, GL_EXP);
    glFogf(GL_FOG_COLOR, (1.0, 1.0, 1.0, 0.0));*/
    glBegin(GL_POINTS);

    for (int i = 0; i < mp_properties->numBody; i++)
    {
        unsigned int red, green, blue;
        getColor(red, green, blue, i);
        glColor3ui(red << 24, green << 24, blue << 24);
        glVertex3f(m_bodies.at(i).position.x / (mp_properties->positionScale * 10), m_bodies.at(i).position.y / (mp_properties->positionScale * 10), m_bodies.at(i).position.z / (mp_properties->positionScale * 10));
    }
    glEnd();

    glutSwapBuffers();

}
