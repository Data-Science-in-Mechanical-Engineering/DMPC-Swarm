/* This is a new task for the emergency stop. If something unexpected happens (like crush...),
** by pulling out the crazyradio can make the system stop.
** Tutorial for adding a new task in Crazyflie System:
** https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/systemtask/
**********************************************************************************************************
** Author: Shengsi Xu
** Date: 20.05.2022
** Version: 0.1 (Test version)
*/


#include "_dsme_emergencystop.h"

#include "config.h"
#include "debug.h"
#include "FreeRTOS.h"
#include "queue.h"
#include "static_mem.h"
#include "task.h"
#include "system.h"

static xQueueHandle watchdogQueue;
static void emStopTask(void *);
static bool isInit = false;
static const portTickType waitTimeInTicks = pdMS_TO_TICKS(500);

// Mem-alloc for the queue and task using the Macro of static_mem.h, which is provided by Crazyflie
STATIC_MEM_QUEUE_ALLOC(watchdogQueue, 5, sizeof(uint8_t));
STATIC_MEM_TASK_ALLOC(emStopTask, EM_STOP_TASK_STACKSIZE);

static void emStopTask(void *param)
{
    DEBUG_PRINT("Emergency Stop function is running!");

    uint8_t heartBeat;
    if (pdTRUE == xQueueReceive(watchdogQueue, &heartBeat, portMAX_DELAY))
    {
        DEBUG_PRINT("First position info received!");
    }

    while (true)
    {
        if (pdFALSE == xQueueReceive(watchdogQueue, &heartBeat, waitTimeInTicks)) // If the Queue is empty, wait 200Ms
        {
            systemRequestShutdown(); // If after 200Ms is the Queue still empty, shutdown.
            DEBUG_PRINT("Emergency Stop request was sent!");
        }
    }
}

void emStopTaskInit()
{
    watchdogQueue = STATIC_MEM_QUEUE_CREATE(watchdogQueue);
    if (!watchdogQueue)
    {
        isInit = false;
        return;
    }

    STATIC_MEM_TASK_CREATE(emStopTask, emStopTask, EM_STOP_TASK_NAME, NULL, EM_STOP_TASK_PRI);
    isInit = true;
}

bool emStopTaskTest()
{
    return isInit;
}

void emStopTaskEnqueueInput(uint8_t heartBeat)
{
    uint8_t _heartBeat = heartBeat;

    xQueueSend(watchdogQueue, (void *)&_heartBeat, portMAX_DELAY);
}