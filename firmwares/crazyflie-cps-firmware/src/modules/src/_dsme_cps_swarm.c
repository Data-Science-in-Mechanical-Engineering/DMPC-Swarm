/* This is a new task for the emergency stop. If something unexpected happens (like crush...),
** by pulling out the crazyradio can make the system stop.
** Tutorial for adding a new task in Crazyflie System:
** https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/systemtask/
**********************************************************************************************************
** Author: Shengsi Xu
** Date: 20.05.2022
** Version: 0.1 (Test version)
*/
#include <string.h>
#include "config.h"
#include "debug.h"
#include "FreeRTOS.h"
#include "task.h"
#include "semphr.h"
#include "queue.h"
#include "static_mem.h"
#include "task.h"
#include "system.h"

#include "commander.h"
#include "estimator.h"

#include "_dsme_trajectory_interpolation.h"
#include "_dsme_cps_swarm.h"

static cps_swarm_state_t cps_state = IDLE;
static bool isInit = false;

static void cpsSwarmTask(void *);

static xQueueHandle mocapQueue;

// Mem-alloc for the queue and task using the Macro of static_mem.h, which is provided by Crazyflie
STATIC_MEM_TASK_ALLOC(cpsSwarmTask, CPS_SWARM_TASK_STACKSIZE);
STATIC_MEM_QUEUE_ALLOC(mocapQueue, 5, sizeof(uint8_t));

static void cpsSwarmTask(void *param)
{
    systemWaitStart();

    // wait till motion capture system is connected.
    uint8_t heartBeat;
    while (pdTRUE != xQueueReceive(mocapQueue, &heartBeat, portMAX_DELAY)){}
    
    cps_state = LAUNCH;
    start_crazyflie(-1.0f, 1.0f, 1.2f);
    vTaskDelay(M2T(10000));
    land_crazyflie();
    while (true) {
    vTaskDelay(M2T(10000));
    }
}

void cpsSwarmTaskInit()
{
    isInit = true;
    mocapQueue = STATIC_MEM_QUEUE_CREATE(mocapQueue);
    STATIC_MEM_TASK_CREATE(cpsSwarmTask, cpsSwarmTask, CPS_SWARM_NAME, NULL, CPS_SWARM_PRI);
}

bool cpsSwarmTaskTest()
{
    return isInit;
}

/*void cpsSwarmMoCapSystemConnectedCallback(uint8_t heartBeat)
{
    uint8_t _heartBeat = heartBeat;

    xQueueSend(mocapQueue, (void *)&_heartBeat, 0);
}*/