/* This is the task to translate the received trajectory to setpoints for the drone
** Tutorial for adding a new task in Crazyflie System:
** https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/development/systemtask/
**********************************************************************************************************
** Author: Alexander Graefe
** Date: 20.06.2022
** Version: 0.1 (Test version)
*/


#ifndef DSME_TRAJECTORY_INTERPOLATION_H
#define DSME_TRAJECTORY_INTERPOLATION_H

#include <stdbool.h>
#include "cf_state_machine.h"

#define TRAJ_INT_SETPOINT_PRIO 2

#define LAUNCH_SPEED 0.2f
#define LAUNCH_STEP_SIZE ((float) LAUNCH_SPEED * ROUND_LENGTH_S / NUM_INT_POINTS_PER_ROUND)

#define NUM_INPUT_POINTS_PER_ROUND 1  
#define PREDICTION_HORIZON 15  // prediction horizon in rounds
#define NUM_STATES 3

#define REACHED_POS(pos, target) ((pos - target < 0.01f) && (target - pos < 0.01f))
#define LAUNCH_STEP(pos, target) (((pos - target < LAUNCH_STEP_SIZE) && (target - pos < LAUNCH_STEP_SIZE)) ? (target - pos) : (LAUNCH_STEP_SIZE * ((target - pos > 0) ? 1.0f : -1.0f)))

typedef enum
 {
     DISABLED,
     TRAJECTORY_FOLLOWING,
     START,
     LAND
 } traj_int_state_t;


/* These two function should be called in the system.c */
void trajIntTaskInit();
bool trajIntTaskTest();

void trajIntNewTrajReceivedCallback(float *new_x_acc, float *new_y_acc, float *new_z_acc, float *new_x_state, float *new_y_state, float *new_z_state);

void calculateNextState();
void round_finished(uint32_t round);

void cp_connected_callback(uint8_t id);
void get_cf_state(float *state, uint32_t round);
uint8_t cf_launch_status();
uint8_t cf_land_status();
void init_position(float *pos, uint8_t id);
void set_current_trajectory(cf_trajectory *ct);
cf_trajectory *get_current_trajectory();
void set_state_directly(float *new_x_state, float *new_y_state, float *new_z_state);
void get_setpoint_state(float *setpoint_x_state, float *setpoint_y_state, float *setpoint_z_state);

void start_crazyflie(float x_start, float y_start, float z_start);
void land_crazyflie();

void interpolate(float *state, float current_input);

#endif