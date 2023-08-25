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
#include "stabilizer.h"

#include "_dsme_trajectory_interpolation.h"
#include "cf_state_machine.h"
#include <math.h>
#include "cfassert.h"

//#define DISABLE_CPS
//#define INIT_POS {{-1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}
/*#define TARGET_POS {{{0.0f, -1.0f, 1.2f}, {-0.3f, 0.0f, 1.0f}, {0.3f, 0.0f, 1.0f}}, \
					{{0.0f, 1.0f, 1.2f}, {-0.3f, 0.0f, 1.0f}, {0.3f, 0.0f, 1.0f}}, \
					{{0.0f, -1.0f, 1.2f}, {0.3f, 0.0f, 1.0f}, {-0.3f, 0.0f, 1.0f}}, \
					{{0.0f, -1.0f, 1.2f}, {-0.3f, 0.0f, 1.0f}, {0.3f, 0.0f, 1.0f}}, \
					{{-1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}}*/
//#define NUM_DRONES 3
//#define NUM_TARGET_POS 5

#define ROUND_LENGTH_S 0.2f
#define NUM_INT_POINTS_PER_ROUND 10  // this has to be an integer multiple of NUM_INPUT_POINTS_PER_ROUND, because we do not really control a drone here, it is just 1
#define INT_INTERVAL_LENGTH_S ((ROUND_LENGTH_S) / NUM_INT_POINTS_PER_ROUND)
#define NUM_INPUT_POINTS_PER_ROUND 1
#define NUM_INT_POINTS_PER_INPUT (NUM_INT_POINTS_PER_ROUND / NUM_INPUT_POINTS_PER_ROUND)
#define PREDICTION_HORIZON 15

#define TICKS_WAIT ((TickType_t) (pdMS_TO_TICKS((int) ((ROUND_LENGTH_S * 1000) / NUM_INT_POINTS_PER_ROUND))))

static uint8_t launch_status = STATUS_IDLE;
static uint8_t land_status = STATUS_FLYING;

static cf_trajectory current_trajectory = {.calculated_by=0, .start_round=0};

static uint32_t round_idx = 0; 	// idx of round in trajectory
static float int_point_idx = 0;  // index of int point in one round. it is at maximum NUM_INT_POINTS_PER_ROUND . THe current time can be calculate as
								 // (current_trajectory.start_round + round_idx) * ROUND_LENGTH_S + int_point_idx*INT_INTERVAL_LENGTH_S

static uint8_t new_trajectory_received = 0;  // wether a new trajectory was received and the state has to be updated

static uint8_t CF_IDX = 0;

static traj_int_state_t traj_int_state = DISABLED;

#ifndef DISABLE_CPS
static setpoint_t motor_stop_setpoint;  // give this setpoint to controller to stop motors (somehow works without initialising it)
#endif
static void trajIntTask(void *);
static bool isInit = false;

static float x_state[NUM_STATES];
static float y_state[NUM_STATES];
static float z_state[NUM_STATES];

static uint8_t state_directly_set = 0;

// initial position of the crazflie when launching (on the ground)
static setpoint_t setpoint;

static uint32_t time_since_last_update;

// Semaphore to signal data is processed and should not be changed
static SemaphoreHandle_t dataSemaphore;

// Mem-alloc for the queue and task using the Macro of static_mem.h, which is provided by Crazyflie
STATIC_MEM_TASK_ALLOC(trajIntTask, TRAJ_INT_TASK_STACKSIZE);

static void trajIntTask(void *param)
{
    systemWaitStart();
    xSemaphoreGive(dataSemaphore);
    uint16_t i = 0;
    while (true) {
        TickType_t start_time = xTaskGetTickCount();
        xSemaphoreTake(dataSemaphore, portMAX_DELAY);
        switch (traj_int_state)
        {
            case DISABLED:
            #ifndef DISABLE_CPS
                //commanderSetSetpoint(&motor_stop_setpoint, TRAJ_INT_SETPOINT_PRIO);
            #endif
                break;
            case START:
            case LAND:
                /*if (i%10 == 0) {
                    state_t current_state;
                    get_state(&current_state);
                    DEBUG_PRINT("%i, %i, %i\n", (int16_t) (setpoint.position.x*1000.0f), (int16_t) (setpoint.position.y*1000), (int16_t) (setpoint.position.z*1000));
                    DEBUG_PRINT("%i, %i, %i\n", (int16_t) (current_state.position.x*1000.0f), (int16_t) (current_state.position.y*1000), (int16_t) (current_state.position.z*1000));
                    if (isnan(current_state.position.x)) {
                        ASSERT(0);
                    }
                    setpoint.position.x = current_state.position.x;
                    DEBUG_PRINT("c\n");
                }*/
                // update setpoint
                {
                setpoint.position.y += (float) LAUNCH_STEP(setpoint.position.y, y_state[0]);
                setpoint.position.z += (float) LAUNCH_STEP(setpoint.position.z, z_state[0]);
                setpoint.position.x += (float) LAUNCH_STEP(setpoint.position.x, x_state[0]);
                ASSERT(!isnan(x_state[0]));
                /*if (isnan(setpoint.position.x)) {
                    if (i%10 == 0) {DEBUG_PRINT("NAN\n");}
                }

                if (i%10 == 0) {
                    DEBUG_PRINT("%i, %i, %i\n", (int16_t) (setpoint.position.y*1000), (int16_t) (setpoint.position.z*1000), (int16_t) (setpoint.position.x*100.0f));
                    DEBUG_PRINT("%i, %i, %i\n", (int16_t) (x_state[0]*100), (int16_t) (y_state[0]*100), (int16_t) (z_state[0]*100));
                }*/
            
                commanderSetSetpoint(&setpoint, TRAJ_INT_SETPOINT_PRIO);
                if (REACHED_POS(setpoint.position.x, x_state[0]) && REACHED_POS(setpoint.position.y, y_state[0]) && REACHED_POS(setpoint.position.z, z_state[0]) ) {
                    if (traj_int_state == START) {
                        setpoint.position.x = x_state[0];
                        setpoint.position.y = y_state[0];
                        setpoint.position.z = z_state[0];
                        traj_int_state = TRAJECTORY_FOLLOWING;
                        launch_status = STATUS_LAUNCHED;
                        land_status = STATUS_FLYING;
                    } else {
                        traj_int_state = DISABLED;
                        land_status = STATUS_LANDED;
                        launch_status = STATUS_IDLE;
                        commanderSetSetpoint(&motor_stop_setpoint, TRAJ_INT_SETPOINT_PRIO);
                    }
                }
                }
                break;
                
            case TRAJECTORY_FOLLOWING:
                if (!state_directly_set) {
                    calculateNextState();
                    state_t current_state;
                    get_state(&current_state);
                    float current_pos[3];
                    current_pos[0] = current_state.position.x;
                    current_pos[1] = current_state.position.y;
                    current_pos[2] = current_state.position.z;

                    float current_setpoint_pos[3];
                    current_setpoint_pos[0] = x_state[0];
                    current_setpoint_pos[1] = y_state[0];
                    current_setpoint_pos[2] = z_state[0];
                    if (CF_DIST(current_pos, current_setpoint_pos) > 0.5f) {
                        //x_state[0] = current_pos[0];
                        //y_state[0] = current_pos[1];
                        //z_state[0] = current_pos[2];

                        //x_state[1] = 0.0f;
                        //y_state[1] = 0.0f;
                        //z_state[1] = 0.0f;

                        //x_state[2] = 0.0f;
                        //y_state[2] = 0.0f;
                        //z_state[2] = 0.0f;

                        for (uint8_t idx = 0; idx < LENGTH_TRAJECTORY; idx++) {
                            //current_trajectory.x_coeff[idx] = 0;
                            //current_trajectory.y_coeff[idx] = 0;
                            //current_trajectory.z_coeff[idx] = 0;
                        }
                    }
                }
                setpoint.position.x = x_state[0];
                setpoint.position.y = y_state[0];
                setpoint.position.z = z_state[0];

                setpoint.velocity.x = x_state[1];
                setpoint.velocity.y = y_state[1];
                setpoint.velocity.z = z_state[1];

                setpoint.acceleration.x = x_state[2];
                setpoint.acceleration.y = y_state[2];
                setpoint.acceleration.z = z_state[2];
                commanderSetSetpoint(&setpoint, TRAJ_INT_SETPOINT_PRIO);
                time_since_last_update++;
                break;
        }
        xSemaphoreGive(dataSemaphore);
        TickType_t delta_t = xTaskGetTickCount() - start_time;
        ASSERT(xTaskGetTickCount() - start_time >= 0);
        if (delta_t < TICKS_WAIT || true) {
            if (i%100 == 0) {
                //DEBUG_PRINT("h\n");
            }
            vTaskDelay(TICKS_WAIT - delta_t);
        }
        if (i%100 == 0) {
            //DEBUG_PRINT("dt: %li\n", delta_t);
            //DEBUG_PRINT("%lu\n", TICKS_WAIT);
            //DEBUG_PRINT("%lu\n", TICKS_WAIT - delta_t);
        }
        i++;
    }
}

void trajIntTaskInit()
{
    isInit = true;

    vSemaphoreCreateBinary(dataSemaphore);

    setpoint.attitude.roll  = 0.0f;
    setpoint.attitude.pitch = 0.0f;
    setpoint.attitude.yaw = 0.0f;
    //setpoint.thrust = 0;
    setpoint.velocity.x = 0.0f;
    setpoint.velocity.y = 0.0f;
    setpoint.velocity.z = 0.0f;

    setpoint.mode.x = modeAbs;
    setpoint.mode.y = modeAbs;
    setpoint.mode.z = modeAbs;
    setpoint.mode.roll = modeAbs;
    setpoint.mode.pitch = modeAbs;
    setpoint.mode.yaw = modeAbs;

    setpoint.attitude.pitch = 0.0f;
    setpoint.attitude.roll = 0.0f;
    setpoint.attitudeRate.pitch = 0.0f;
    setpoint.attitudeRate.roll = 0.0f;


    for (int i = 0; i < NUM_STATES; i++) {
        x_state[i] = 0.0f;
        y_state[i] = 0.0f;
        z_state[i] = 0.0f;
    }
    STATIC_MEM_TASK_CREATE(trajIntTask, trajIntTask, TRAJ_INT_NAME, NULL, TRAJ_INT_PRI);
}

void start_crazyflie(float x_start, float y_start, float z_start)
{
    DEBUG_PRINT("Starting CF\n");
    if (traj_int_state == DISABLED) {
        xSemaphoreTake(dataSemaphore, portMAX_DELAY);
        state_t current_state;
        get_state(&current_state);
        setpoint.position.x = current_state.position.x;
        setpoint.position.y = current_state.position.y;
        setpoint.position.z = current_state.position.z + 0.05f;  // a short boost to get away from ground quickly
        if (isnan(current_state.position.x)) {
                        DEBUG_PRINT("NAN2\n");
                    }

        if (isnan(current_state.position.y)) {
                        DEBUG_PRINT("NAN2\n");
                    }

        x_state[0] = x_start;
        y_state[0] = y_start;
        z_state[0] = z_start;

        traj_int_state = START;
        launch_status = STATUS_LAUNCHING;
        xSemaphoreGive(dataSemaphore);
    }
}

void land_crazyflie() 
{
    xSemaphoreTake(dataSemaphore, portMAX_DELAY);
    x_state[0] = setpoint.position.x;
    y_state[0] = setpoint.position.y;
    z_state[0] = 0.8f;

    traj_int_state = LAND;
    land_status = STATUS_LANDING;
    xSemaphoreGive(dataSemaphore);
}

void interpolate(float *state, float current_input)
{
	float pos = state[0];
	float vel = state[1];
	float acc = state[2];
	state[0] = pos + (INT_INTERVAL_LENGTH_S) * vel + (0.5f * INT_INTERVAL_LENGTH_S * INT_INTERVAL_LENGTH_S) * acc +
		(1/(2.0f*3.0f) * INT_INTERVAL_LENGTH_S * INT_INTERVAL_LENGTH_S * INT_INTERVAL_LENGTH_S) * current_input;
	state[1] = vel + (INT_INTERVAL_LENGTH_S) * acc + (0.5f * INT_INTERVAL_LENGTH_S * INT_INTERVAL_LENGTH_S) * current_input;
	state[2] = acc + (INT_INTERVAL_LENGTH_S) * current_input;
}

void calculateNextState()
{
	uint32_t input_idx = round_idx * NUM_INPUT_POINTS_PER_ROUND + (int_point_idx / NUM_INT_POINTS_PER_INPUT);
	// only update states, if the trajectory is not at its end.
	if (input_idx < PREDICTION_HORIZON * NUM_INPUT_POINTS_PER_ROUND) {
		interpolate(x_state, current_trajectory.x_coeff[input_idx]);
		interpolate(y_state, current_trajectory.y_coeff[input_idx]);
		interpolate(z_state, current_trajectory.z_coeff[input_idx]);
	} else {
        x_state[1] = 0.0f;
        x_state[2] = 0.0f;

        y_state[1] = 0.0f;
        y_state[2] = 0.0f;

        z_state[1] = 0.0f;
        z_state[2] = 0.0f;
    }
	int_point_idx++;
}

void round_finished(uint32_t round)
{
	round_idx = round - current_trajectory.start_round;
	int_point_idx = 0;
	// if we receive a new trajectory, the init state is the current state, because we only update the trajectory,
	// when it is a new one. Because the only new trajectory comes from a CU, the init state is equal to the current state.
	if (new_trajectory_received) {
		for (uint8_t i = 0; i < 3; i++) {
			x_state[i] = current_trajectory.x_state_init[i];
		}
		for (uint8_t i = 0; i < 3; i++) {
			y_state[i] = current_trajectory.y_state_init[i];
		}
		for (uint8_t i = 0; i < 3; i++) {
			z_state[i] = current_trajectory.z_state_init[i];
		}
		new_trajectory_received = 0;
	}
}

uint8_t cf_launch_status()
{
	return launch_status;
}

uint8_t cf_land_status()
{
	return land_status;
}

void init_position(float *pos, uint8_t id)
{
	float init_pos[][3] = INIT_POS;
	pos[0] = init_pos[id][0];
	pos[1] = init_pos[id][1];
	pos[2] = init_pos[id][2];
}

void set_current_trajectory(cf_trajectory *ct)
{
    xSemaphoreTake(dataSemaphore, portMAX_DELAY);
	new_trajectory_received = 1;
	memcpy(&current_trajectory, ct, sizeof(cf_trajectory));
    if (traj_int_state == START) {
        traj_int_state = TRAJECTORY_FOLLOWING;
    }
    xSemaphoreGive(dataSemaphore);
}

cf_trajectory *get_current_trajectory()
{
	return &current_trajectory;
}

void get_setpoint_state(float *setpoint_x_state, float *setpoint_y_state, float *setpoint_z_state)
{
    memcpy(setpoint_x_state, x_state, 3*sizeof(float));
    memcpy(setpoint_y_state, y_state, 3*sizeof(float));
    memcpy(setpoint_z_state, z_state, 3*sizeof(float));
}

void set_state_directly(float *new_x_state, float *new_y_state, float *new_z_state) 
{
    memcpy(x_state, new_x_state, 3*sizeof(float));
    memcpy(y_state, new_y_state, 3*sizeof(float));
    memcpy(z_state, new_z_state, 3*sizeof(float));
    state_directly_set = 1;
}

void cp_connected_callback(uint8_t id)
{
	CF_IDX = id - 1;
	float init_pos[NUM_DRONES][3] = INIT_POS;
	x_state[0] = init_pos[CF_IDX][0];
	y_state[0] = init_pos[CF_IDX][1];
	z_state[0] = init_pos[CF_IDX][2];
}

void get_cf_state(float *state)
{
    xSemaphoreTake(dataSemaphore, portMAX_DELAY);
    state_t current_state;
    get_state(&current_state);
    xSemaphoreGive(dataSemaphore);

    float current_x_state[3];
    float current_y_state[3];
    float current_z_state[3];

    memcpy(current_x_state, x_state, 3*sizeof(float));
    memcpy(current_y_state, y_state, 3*sizeof(float));
    memcpy(current_z_state, z_state, 3*sizeof(float));

    current_x_state[0] = current_state.position.x;
    current_y_state[0] = current_state.position.y;
    current_z_state[0] = current_state.position.z;

    //ASSERT(current_state.position.z > 0.8f);
	
    // simulate state one round forward
    for (uint8_t i = 0; i < NUM_INT_POINTS_PER_ROUND; i++) {
        uint32_t input_idx = round_idx * NUM_INPUT_POINTS_PER_ROUND + ((int_point_idx+i) / NUM_INT_POINTS_PER_INPUT);
        // only update states, if the trajectory is not at its end.
        if (input_idx < PREDICTION_HORIZON * NUM_INPUT_POINTS_PER_ROUND) {
            interpolate(current_x_state, current_trajectory.x_coeff[input_idx]);
            interpolate(current_y_state, current_trajectory.y_coeff[input_idx]);
            interpolate(current_z_state, current_trajectory.z_coeff[input_idx]);
        } else {
            current_x_state[1] = 0.0f;
            current_x_state[2] = 0.0f;

            current_y_state[1] = 0.0f;
            current_y_state[2] = 0.0f;

            current_z_state[1] = 0.0f;
            current_z_state[2] = 0.0f;
        }
    }
    
    state[0] = current_x_state[0]; //current_state.position.x;
	state[1] = current_y_state[0]; //current_state.position.y;
	state[2] = current_z_state[0]; //current_state.position.z;

	state[3] = 0;
	state[4] = 0;
	state[5] = 0;

	state[6] = 0;
	state[7] = 0;
	state[8] = 0;
}

bool trajIntTaskTest()
{
    return isInit;
}