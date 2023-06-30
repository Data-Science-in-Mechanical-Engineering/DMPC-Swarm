#ifndef DSME_CPS_STATEMACHINE_H
#define DSME_CPS_STATEMACHINE_H

typedef enum{
    IDLE,
    AP_ACK,
    ALL_AP_READY,
    LAUNCH,
    RUN,
    FINISH
} cps_swarm_state_t;

void cpsSwarmTaskInit();

bool cpsSwarmTaskTest();

//void cpsSwarmMoCapSystemConnectedCallback(uint8_t heartBeat);

#endif