/*
 * cf_state_machine.h
 *
 *  Created on: 07.11.2022
 *      Author: mf724021
 */

#ifndef INC_CF_STATE_MACHINE_H_
#define INC_CF_STATE_MACHINE_H_

#include "internal_messages.h"

#define DYNAMIC_SWARM 

#define CIRCLE_RADIUS 1.2f
#define ROTATE_CCW 0

// defines for the target positions to fly.
//#define INIT_POS {{-1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}
/*#define TARGET_POS {{{0.0f, -1.0f, 1.0f}, {-0.3f, 0.0f, 1.0f}, {0.3f, 0.0f, 1.0f}}, \
					{{0.0f, 1.0f, 1.0f}, {-0.3f, 0.0f, 1.0f}, {0.3f, 0.0f, 1.0f}}, \
					{{0.0f, -1.0f, 1.0f}, {0.3f, 0.0f, 1.0f}, {-0.3f, 0.0f, 1.0f}}, \
					{{0.0f, -1.0f, 1.0f}, {-0.3f, 0.0f, 1.0f}, {0.3f, 0.0f, 1.0f}}, \
					{{-1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}}*/
#define INIT_POS {{-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}, {-1.0f, -1.0f, 1.0f}, {1.0f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}
#define TARGET_POS {{{-0.6f, 1.0f, 1.0f}, {0.6f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {-0.6f, -1.0f, 1.0f}, {0.6f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}, \
					{{-0.6f, -1.0f, 1.0f}, {0.6f, -1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {-0.6f, 1.0f, 1.0f}, {0.6f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}, \
					{{-0.6f, 1.0f, 1.0f}, {0.6f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, {0.0f, -1.0f, 1.0f}, {-0.6f, -1.0f, 1.0f}, {0.6f, -1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}, {-1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f}}, \
					INIT_POS \
					}

#define NUM_DRONES 10
#define NUM_TARGET_POS 4

#define STATUS_IDLE 0
#define STATUS_LAUNCHING 1
#define STATUS_LAUNCHED 2
#define STATUS_FLYING 3
#define STATUS_CURRENT_TARGET_REACHED 4
#define STATUS_ALL_TARGETS_REACHED 5
#define STATUS_LANDING 6
#define STATUS_LANDED 7

#define START_FROM_HAND 0

#define CF_ABS(X) ((X)>0 ? (X):-(X))
#define CF_MAX(X, Y) (((X)>(Y)) ? (X):(Y))
#define CF_DIST(X, Y) (CF_MAX(CF_MAX(CF_ABS(X[0]-Y[0]), CF_ABS(X[1]-Y[1])), CF_ABS(X[2]-Y[2])))

// defines for quantization from float to uint16_t or backwards
#define MAX_INPUT 5.0f

#define MAX_POSITION 5.0f
#define MAX_VELOCITY 5.0f
#define MAX_ACCELERATION 5.0f

#define ROUND_VALUE(X) (((X) - (int) (X)) > 0.5f ? ((X) + 1) : (X))
#define INT16_MAX_ ((((uint32_t) 1)<<15) - 1)

#define DEQUANTIZE(X, MAX_VAL) (((float) (((float) (X)) - INT16_MAX_) / INT16_MAX_) * MAX_VAL)  //((MAX_VAL*2.0f/((((uint32_t) 1)<<16) - 1)) * X - MAX_VAL)
#define DEQUANTIZE_INPUT(X) DEQUANTIZE(X, MAX_INPUT)
#define DEQUANTIZE_POSITION(X) DEQUANTIZE(X, MAX_POSITION)
#define DEQUANTIZE_VELOCITY(X) DEQUANTIZE(X, MAX_VELOCITY)
#define DEQUANTIZE_ACCELERATION(X) DEQUANTIZE(X, MAX_ACCELERATION)

#define QUANTIZE(X, MAX_VAL) ((uint16_t) (ROUND_VALUE((X) / (MAX_VAL) * INT16_MAX_) + INT16_MAX_))   //((uint16_t) ((X + MAX_VAL)/(2*MAX_VAL)*((float) (((uint32_t) 1)<<16) - 1)))
#define QUANTIZE_POSITION(X) QUANTIZE(X, MAX_POSITION)
#define QUANTIZE_VELOCITY(X) QUANTIZE(X, MAX_VELOCITY)
#define QUANTIZE_ACCELERATION(X) QUANTIZE(X, MAX_ACCELERATION)
#define QUANTIZE_INPUT(X) QUANTIZE(X, MAX_INPUT)

typedef enum cf_state_tag {
    IDLE_STATE,
    AP_READY_STATE,
	SYS_READY_STATE,
    LAUNCH_STATE,
    SYS_RUN_STATE,
	SYNC_MOVEMENT_STATE,
    SYS_SHUTDOWN_STATE,
	WAIT_FOR_LAUNCH_STATE,
	VERTICAL_LAUNCH_STATE
} cf_state;

typedef struct cf_trajectory_tag
{
	float x_coeff[LENGTH_TRAJECTORY];
	float y_coeff[LENGTH_TRAJECTORY];
	float z_coeff[LENGTH_TRAJECTORY];

	float x_state_init[3];
	float y_state_init[3];
	float z_state_init[3];

	uint32_t start_round;

	uint8_t calculated_by;
} cf_trajectory;

typedef struct cf_state_machine_handle_tag
{
	cf_state state;
	ap_com_handle hap_com;
	uint8_t id;
	uint8_t num_drones;
	float round_length_s;
	trajectory_message_t current_traj;
	uint8_t target_position_idx;      // current index of target position
	uint16_t current_target_quant[3]; //save this, such that we do not have to calculate it everytime.
	float current_target[3];
	float current_target_angle;
	void (*round_finished)();
	void (*wait_cf_to_start)();
	void (*cp_connected_callback)(uint8_t);
	void (*get_cf_state)(float *);
	void (*launch_cf)(float, float, float);
	void (*land_cf)();
	uint8_t (*cf_launch_status)();
	uint8_t (*cf_land_status)();
	void (*init_position)(float *, uint8_t);
	cf_trajectory *(*get_current_trajectory)();  // returns a pointer on the trajectory.
	void (*set_current_trajectory)(cf_trajectory *); // sets a new trajectory of the crazyflie
	void (*set_state_directly)(float *, float *, float *);
	void (*get_setpoint_state)(float *, float *, float *);
	bool (*mocap_system_active)();
} cf_state_machine_handle;


/**
 * inits the statmachine
 * @param hstate_machine handle to state machine
 * @param wait_cf_to_start function polling till cf started
 * @param get_cf_state writes state of the cf into its parameter
 * @param launch_cf launches the crazyflie to the given x, y and z coordinates
 * @param land_cf lands cf
 * @param cf_launch_status gives the launch status of the crazyflie (STATUS_LAUNCHING, STATUS_LAUNCHED, STATUS_IDLE)
 * @param cf_land_status gives the land status of the crazyflie (STATUS_LANDING, STATUS_LANDED, STATUS_FLYING)
 * @param init_position returns the init position of the cf
 * @param get_current_trajectory returns a pointer on the trajectory. It is not const., the state machine might write new data into it.
 * @param ap_communicate_p function for communicating with cp (or ap)
 * @param com_wait_p function which waits a small amount of time between the ap communication.
 */
void init_cf_state_machine(cf_state_machine_handle *hstate_machine,
							void (*round_finished)(uint32_t),
							void (*wait_cf_to_start)(),
							void (*cp_connected_callback)(uint8_t),
							void (*get_cf_state)(float *),
							void (*launch_cf)(float, float, float),
							void (*land_cf)(),
							uint8_t (*cf_launch_status)(),
							uint8_t (*cf_land_status)(),
							void (*init_position)(float *, uint8_t),
							cf_trajectory *(*get_current_trajectory)(),
							void (*set_current_trajectory)(cf_trajectory *),
							void (*ap_send_p)(uint8_t *, uint16_t),
							void (*ap_receive_p)(uint8_t *, uint16_t),
							void (*com_wait_p)(),
							void (*set_state_directly)(float *, float *, float *),
							void (*get_setpoint_state)(float *, float *, float *),
							bool (*mocap_system_active)());

/**
 * runs the state machine
 */
void run_cf_state_machine(cf_state_machine_handle *hstate_machine);

#endif /* INC_CF_STATE_MACHINE_H_ */
