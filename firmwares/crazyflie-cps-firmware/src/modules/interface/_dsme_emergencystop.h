/* This is a new task for the emergency stop. If something unexpected happens (like crush...),
** by pulling out the crazyradio can make the system stop.
** Tutorial for adding a new task in Crazyflie System:
** https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/systemtask/
**********************************************************************************************************
** Author: Shengsi Xu
** Date: 20.05.2022
** Version: 0.1 (Test version)
*/

#ifndef __DSME_EMERGENCYSTOP__
#define __DSME_EMERGENCYSTOP__

#include <stdbool.h>
#include <stdint.h>

/* These two function should be called in the system.c */
void emStopTaskInit();
bool emStopTaskTest();

/* This function */
void emStopTaskEnqueueInput(uint8_t heartBeat);

#endif //_dsme_emergencystop.h