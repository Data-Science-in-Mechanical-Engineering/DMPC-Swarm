#ifndef CPS_CONFIG_H
#define CPS_CONFIG_H

#include <stdint.h>
//
static const uint8_t drones_ids[] = {1, 2, 3, 4, 5, 6};
// define initial positions of agents position belongs to agent id with same index in drone_ids
static const float drones_init_pos[3][3] = {{-1.0f, 1.0f, 1.2f}, {0.0f, 0.0f, 1.2f}, {1.0f, 0.0f, 1.2f}};
#endif