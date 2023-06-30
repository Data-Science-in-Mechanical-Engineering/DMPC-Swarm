#ifndef __DSME_SPI_SLAVE_H__
#define __DSME_SPI_SLAVE_H__

#include <stdbool.h>
#include "internal_messages.h"

#define MAX_INPUT 5.0f 

#define MAX_POSITION 5.0f
#define MAX_VELOCITY 5.0f
#define MAX_ACCELERATION 5.0f

#define DEQUANTIZE(X, MAX_VAL) ((MAX_VAL*2.0f/((((uint32_t) 1)<<16) - 1)) * X - MAX_VAL)
#define DEQUANTIZE_INPUT(X) DEQUANTIZE(X, MAX_INPUT)
#define DEQUANTIZE_POSITION(X) DEQUANTIZE(X, MAX_POSITION)
#define DEQUANTIZE_VELOCITY(X) DEQUANTIZE(X, MAX_VELOCITY)
#define DEQUANTIZE_ACCELERATION(X) DEQUANTIZE(X, MAX_ACCELERATION)

void spiSlaveTaskInit();
bool spiSlaveTaskTest();

void spiSlaveMoCapSystemConnectedCallback(uint8_t heartBeat);

bool mocapSystemActive();

//void get_init_pos(uint8_t id, float *init_pos);

#endif