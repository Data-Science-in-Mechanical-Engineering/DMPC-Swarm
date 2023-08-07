/*
 * cf_state_machine.c
 *
 *  Created on: 07.11.2022
 *      Author: mf724021
 */
#include "FreeRTOS.h"
#include "queue.h"
#include "semphr.h"
#include "task.h"
#include "cf_state_machine.h"
#include "internal_messages.h"
#include <stdio.h>

#include "debug.h"
#include "math.h"

#if START_FROM_HAND
static uint32_t was_low = 0;
#endif
static uint32_t wait_for_launch = 0;

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
							bool (*mocap_system_active)())
{
	hstate_machine->state = IDLE_STATE;
	hstate_machine->round_length_s = -1.0f;
	hstate_machine->id = (uint8_t) -1u;
	hstate_machine->round_finished = round_finished;
	hstate_machine->wait_cf_to_start = wait_cf_to_start;
	hstate_machine->cp_connected_callback = cp_connected_callback;
	hstate_machine->get_cf_state = get_cf_state;
	hstate_machine->launch_cf = launch_cf;
	hstate_machine->land_cf = land_cf;
	hstate_machine->cf_launch_status = cf_launch_status;
	hstate_machine->cf_land_status = cf_land_status;
	hstate_machine->init_position = init_position;
	hstate_machine->get_current_trajectory = get_current_trajectory;
	hstate_machine->set_current_trajectory = set_current_trajectory;
	hstate_machine->set_state_directly = set_state_directly;
	hstate_machine->get_setpoint_state = get_setpoint_state;
	hstate_machine->mocap_system_active = mocap_system_active;
	init_ap_com(&hstate_machine->hap_com, ap_send_p, ap_receive_p, com_wait_p, com_wait_p);

	hstate_machine->wait_cf_to_start();
}


/**
 * @brief dequantizes the message received over the network into a real trajectory.
 */
static void message_to_trajectory(cf_state_machine_handle *hstate_machine, trajectory_message_t *trajectory_message, cf_trajectory *current_traj)
{
	for (uint16_t i = 0; i < LENGTH_TRAJECTORY; i++) {
		current_traj->x_coeff[i] = DEQUANTIZE_INPUT(trajectory_message->trajectory[i*3]);
		current_traj->y_coeff[i] = DEQUANTIZE_INPUT(trajectory_message->trajectory[i*3+1]);
		current_traj->z_coeff[i] = DEQUANTIZE_INPUT(trajectory_message->trajectory[i*3+2]);
	}

	current_traj->x_state_init[0] = DEQUANTIZE_POSITION(trajectory_message->init_state[0]);
	current_traj->y_state_init[0] = DEQUANTIZE_POSITION(trajectory_message->init_state[1]);
	current_traj->z_state_init[0] = DEQUANTIZE_POSITION(trajectory_message->init_state[2]);

	current_traj->x_state_init[1] = DEQUANTIZE_VELOCITY(trajectory_message->init_state[3]);
	current_traj->y_state_init[1] = DEQUANTIZE_VELOCITY(trajectory_message->init_state[4]);
	current_traj->z_state_init[1] = DEQUANTIZE_VELOCITY(trajectory_message->init_state[5]);

	current_traj->x_state_init[2] = DEQUANTIZE_ACCELERATION(trajectory_message->init_state[6]);
	current_traj->y_state_init[2] = DEQUANTIZE_ACCELERATION(trajectory_message->init_state[7]);
	current_traj->z_state_init[2] = DEQUANTIZE_ACCELERATION(trajectory_message->init_state[8]);

	current_traj->calculated_by = trajectory_message->calculated_by;
	current_traj->start_round = trajectory_message->trajectory_start_time;
}

static void quantize_state(state_message_t *message, float *state)
{
	message->state[0] = QUANTIZE_POSITION(state[0]);
	message->state[1] = QUANTIZE_POSITION(state[1]);
	message->state[2] = QUANTIZE_POSITION(state[2]);

	message->state[3] = QUANTIZE_VELOCITY(state[3]);
	message->state[4] = QUANTIZE_VELOCITY(state[4]);
	message->state[5] = QUANTIZE_VELOCITY(state[5]);

	message->state[6] = QUANTIZE_ACCELERATION(state[6]);
	message->state[7] = QUANTIZE_ACCELERATION(state[7]);
	message->state[8] = QUANTIZE_ACCELERATION(state[8]);
}



static void set_new_target_pos(cf_state_machine_handle *hstate_machine)
{
	#ifndef DYNAMIC_SWARM
	float target_pos[][NUM_DRONES][3] = TARGET_POS;
	hstate_machine->current_target_quant[0] = QUANTIZE_POSITION(target_pos[hstate_machine->target_position_idx][hstate_machine->id-1][0]);
	hstate_machine->current_target_quant[1] = QUANTIZE_POSITION(target_pos[hstate_machine->target_position_idx][hstate_machine->id-1][1]);
	hstate_machine->current_target_quant[2] = QUANTIZE_POSITION(target_pos[hstate_machine->target_position_idx][hstate_machine->id-1][2]);

	hstate_machine->current_target[0] = target_pos[hstate_machine->target_position_idx][hstate_machine->id-1][0];
	hstate_machine->current_target[1] = target_pos[hstate_machine->target_position_idx][hstate_machine->id-1][1];
	hstate_machine->current_target[2] = target_pos[hstate_machine->target_position_idx][hstate_machine->id-1][2];
	#else
	float target_pos[3];
	float offset = (hstate_machine->id-1) * 3.1415926535f / 4;
	target_pos[0] = (CIRCLE_RADIUS)*cosf(hstate_machine->current_target_angle + offset);
	target_pos[1] = (CIRCLE_RADIUS)*sinf(hstate_machine->current_target_angle + offset);
	target_pos[2] = 1.0f+ hstate_machine->id*0.00f;
	hstate_machine->current_target_quant[0] = QUANTIZE_POSITION(target_pos[0]);
	hstate_machine->current_target_quant[1] = QUANTIZE_POSITION(target_pos[1]);
	hstate_machine->current_target_quant[2] = QUANTIZE_POSITION(target_pos[2]);

	hstate_machine->current_target[0] = target_pos[0];
	hstate_machine->current_target[1] = target_pos[1];
	hstate_machine->current_target[2] = target_pos[2];
	#endif
}

static void state_message_write_target_pos(state_message_t *state_message, cf_state_machine_handle *hstate_machine)
{
	state_message->target_position_idx = hstate_machine->target_position_idx;
	state_message->current_target[0] = hstate_machine->current_target_quant[0];
	state_message->current_target[1] = hstate_machine->current_target_quant[1];
	state_message->current_target[2] = hstate_machine->current_target_quant[2];
}

static void state_message_write_traj_metadata(state_message_t *state_message, cf_state_machine_handle *hstate_machine)
{
	cf_trajectory *traj = hstate_machine->get_current_trajectory();
	state_message->calculated_by = traj->calculated_by;
	state_message->trajectory_start_time = traj->start_round;
}

/**
 * @returns length of data in tx_data
 */
static uint16_t process_IDLE_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// do nothing if size is not 1
	if (size == 1) {
		if (rx_data[0]->header.type == TYPE_METADATA) {
			hstate_machine->id = rx_data[0]->header.id;
			hstate_machine->state = AP_READY_STATE;
			hstate_machine->num_drones = rx_data[0]->metadata_message.num_drones;
			hstate_machine->round_length_s = rx_data[0]->metadata_message.round_length_ms / 1000.0f;
			tx_data->header.type = TYPE_AP_ACK;
			tx_data->header.id = hstate_machine->id;

			// set current target to first one
			hstate_machine->target_position_idx = 0;
			set_new_target_pos(hstate_machine);

			// tell cf that it is now connected to the CP (currently only used fr the simulator)
			hstate_machine->cp_connected_callback(hstate_machine->id);

			#ifndef DYNAMIC_SWARM
			// init the last field of hstate_machine, the current trajectory. This field is only needed, if the CU requests the trajectory.
			// it is saved in the statemachine and not requested evertime, because then, we need to do all the quantizations again.
			hstate_machine->current_traj.header.type = TYPE_TRAJECTORY;
			for (uint16_t i = 0; i < LENGTH_TRAJECTORY*3; i++) {
				hstate_machine->current_traj.trajectory[i] = QUANTIZE_INPUT(0.0f);
			}
			float state[3];
			hstate_machine->init_position(state, hstate_machine->id-1);
			// somehow the z-value is wrong when the drone sits at the ground. (if the mocap system is not started yet)
			//hstate_machine->get_cf_state(state);
			hstate_machine->current_traj.init_state[0] = QUANTIZE_POSITION(state[0]);
			hstate_machine->current_traj.init_state[1] = QUANTIZE_POSITION(state[1]);
			hstate_machine->current_traj.init_state[2] = QUANTIZE_POSITION(state[2]);

			hstate_machine->current_traj.init_state[3] = QUANTIZE_VELOCITY(0.0f);
			hstate_machine->current_traj.init_state[4] = QUANTIZE_VELOCITY(0.0f);
			hstate_machine->current_traj.init_state[5] = QUANTIZE_VELOCITY(0.0f);

			hstate_machine->current_traj.init_state[6] = QUANTIZE_ACCELERATION(0.0f);
			hstate_machine->current_traj.init_state[7] = QUANTIZE_ACCELERATION(0.0f);
			hstate_machine->current_traj.init_state[8] = QUANTIZE_ACCELERATION(0.0f);

			hstate_machine->current_traj.calculated_by = 0;
			hstate_machine->current_traj.trajectory_start_time = 0;
			hstate_machine->current_traj.drone_id = hstate_machine->id;
			#endif
			#ifdef DYNAMIC_SWARM
			hstate_machine->state = WAIT_FOR_LAUNCH_STATE;
			wait_for_launch = 0;
			#endif
			return 1;
		}
	}
	tx_data->header.type = TYPE_ERROR;
	tx_data->header.id = hstate_machine->id;
	return 1;
}

static uint16_t process_WAIT_FOR_LAUNCH_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// wait for the UAV to be lifted off the ground
	float state[9];
	hstate_machine->get_cf_state(state);

#if START_FROM_HAND
	if (state[2] < 0.1f) {
		was_low += 1;
	}
	if (state[2] > 1.0f && (hstate_machine->mocap_system_active()) && was_low > 50) {
		hstate_machine->state = VERTICAL_LAUNCH_STATE;

	}
#else
	wait_for_launch += 1;
	if (hstate_machine->mocap_system_active() && wait_for_launch > 5) {
		hstate_machine->state = VERTICAL_LAUNCH_STATE;

	}
#endif
	// dont send anything yet.
	tx_data->header.type = TYPE_DUMMY;
	tx_data->header.id = hstate_machine->id;
	return 1;
}

static uint16_t process_VERTICAL_LAUNCH_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// wait for the UAV to be lifted off the ground
	float state[9];
	hstate_machine->get_cf_state(state);

	uint8_t status = hstate_machine->cf_launch_status();
	if (status == STATUS_IDLE) {
		hstate_machine->launch_cf(state[0], state[1], 1.0f+hstate_machine->id*0.00f);

		// init the last field of hstate_machine, the current trajectory. This field is only needed, if the CU requests the trajectory.
		// it is saved in the statemachine and not requested evertime, because then, we need to do all the quantizations again.
		hstate_machine->current_traj.header.type = TYPE_TRAJECTORY;
		for (uint16_t i = 0; i < LENGTH_TRAJECTORY*3; i++) {
			hstate_machine->current_traj.trajectory[i] = QUANTIZE_INPUT(0.0f);
		}
		hstate_machine->current_traj.init_state[0] = QUANTIZE_POSITION(state[0]);
		hstate_machine->current_traj.init_state[1] = QUANTIZE_POSITION(state[1]);
		hstate_machine->current_traj.init_state[2] = QUANTIZE_POSITION(1.0f+hstate_machine->id*0.00f);

		hstate_machine->current_traj.init_state[3] = QUANTIZE_VELOCITY(0.0f);
		hstate_machine->current_traj.init_state[4] = QUANTIZE_VELOCITY(0.0f);
		hstate_machine->current_traj.init_state[5] = QUANTIZE_VELOCITY(0.0f);

		hstate_machine->current_traj.init_state[6] = QUANTIZE_ACCELERATION(0.0f);
		hstate_machine->current_traj.init_state[7] = QUANTIZE_ACCELERATION(0.0f);
		hstate_machine->current_traj.init_state[8] = QUANTIZE_ACCELERATION(0.0f);

		hstate_machine->current_traj.calculated_by = 0;
		hstate_machine->current_traj.trajectory_start_time = 0;
		hstate_machine->current_traj.drone_id = hstate_machine->id;

		hstate_machine->current_target_angle = 1.5f*3.1415926535f;
	}
	if (status == STATUS_LAUNCHED) {
		hstate_machine->state = SYS_RUN_STATE;
	}

	// dont send anything yet.
	tx_data->header.type = TYPE_DUMMY;
	tx_data->header.id = hstate_machine->id;
	return 1;
}

/**
 * @returns length of data in tx_data
 */
static uint16_t process_AP_READY_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// iterate through array
	for (uint16_t i = 0; i < size; i++) {
		switch (rx_data[i]->header.type)
		{
			case TYPE_DRONE_STATE:
				break;
			case TYPE_TRAJECTORY:
				break;
			case TRANSFORM_TYPE(TYPE_LAUNCH_DRONES):
				break;
			// if CF receives this from the CP somehow the CP restarted.
			case TYPE_METADATA:
			case TYPE_CP_ACK:
				hstate_machine->id = rx_data[i]->header.id;
				hstate_machine->state = SYS_READY_STATE;
				tx_data->header.type = TYPE_AP_ACK;
				tx_data->header.id = hstate_machine->id;
				return 1;
			default:
				break;
		}
	}
	float state[9];
	hstate_machine->get_cf_state(state);
	tx_data->header.type = TYPE_DRONE_STATE;
	tx_data->header.id = hstate_machine->id;
	quantize_state(&tx_data->state_message, state);
	tx_data->state_message.status = STATUS_IDLE;
	state_message_write_target_pos((state_message_t *) tx_data, hstate_machine);
	state_message_write_traj_metadata((state_message_t *) tx_data, hstate_machine);
	return 1;
}

/**
 * @returns length of data in tx_data
 */
static uint16_t process_SYS_READY_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// iterate through array
	for (uint16_t i = 0; i < size; i++) {
		switch (rx_data[i]->header.type)
		{
			case TYPE_DRONE_STATE:
				break;
			case TYPE_TRAJECTORY:
				break;
			case TYPE_LAUNCH_DRONES:
				hstate_machine->state = LAUNCH_STATE;
				break;
			default:
				break;
		}
	}
	float state[9];
	hstate_machine->get_cf_state(state);
	tx_data->header.type = TYPE_DRONE_STATE;
	tx_data->header.id = hstate_machine->id;
	quantize_state(&tx_data->state_message, state);
	tx_data->state_message.status = STATUS_IDLE;
	state_message_write_target_pos((state_message_t *) tx_data, hstate_machine);
	state_message_write_traj_metadata((state_message_t *) tx_data, hstate_machine);
	return 1;
}

static uint16_t process_LAUNCH_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// iterate through array
	uint16_t num_launched_drones = 0;
	uint8_t land_drones = 0;
	for (uint16_t i = 0; i < size; i++) {
		switch (rx_data[i]->header.type)
		{
			case TYPE_DRONE_STATE:
				// count how many drones have launched
				if (rx_data[i]->state_message.status == STATUS_LAUNCHED) {
					num_launched_drones++;
				}
				break;
			case TYPE_SYS_SHUTDOWN:
				land_drones = 1;
				break;
			// ignore others
			default:
				break;
		}
	}

	uint8_t status = hstate_machine->cf_launch_status();
	if (status == STATUS_IDLE) {
		float init_states[3];
		hstate_machine->init_position(init_states, hstate_machine->id-1);
		hstate_machine->launch_cf(init_states[0], init_states[1], init_states[2]);
	}
	float state[9];
	hstate_machine->get_cf_state(state);
	tx_data->header.type = TYPE_DRONE_STATE;
	tx_data->header.id = hstate_machine->id;
	quantize_state(&tx_data->state_message, state);
	tx_data->state_message.status = status;
	state_message_write_target_pos((state_message_t *) tx_data, hstate_machine);
	state_message_write_traj_metadata((state_message_t *) tx_data, hstate_machine);

	if (num_launched_drones == hstate_machine->num_drones) {
		hstate_machine->state = SYS_RUN_STATE;
	}
	if (land_drones) {
		hstate_machine->state = SYS_SHUTDOWN_STATE;
	}
	return 1;

}

static uint16_t process_SYS_RUN_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data, uint32_t *round_nmbr)
{
	// iterate through array
	uint16_t num_current_targets_reached = 0;
	uint8_t target_changed = 0;  // if we miss some messages the other drones might already have changed to the next target position. 
	uint8_t land_drones = 0;
	//uint8_t updated = 0;
	cf_trajectory *current_traj;
	for (uint16_t i = 0; i < size; i++) {
		switch (rx_data[i]->header.type)
		{
			case TYPE_DRONE_STATE:
				#ifndef DYNAMIC_SWARM
				// count how many drones have launched
				if ((rx_data[i]->state_message.status == STATUS_CURRENT_TARGET_REACHED || rx_data[i]->state_message.status == STATUS_ALL_TARGETS_REACHED) && hstate_machine->target_position_idx == rx_data[i]->state_message.target_position_idx) {
					if (rx_data[i]->header.id != hstate_machine->id) {
						num_current_targets_reached++;
					}
				}
				// check if the other drone is already flying to the next target position
				if (rx_data[i]->state_message.target_position_idx > hstate_machine->target_position_idx) {
					hstate_machine->target_position_idx = rx_data[i]->state_message.target_position_idx;
					if (hstate_machine->target_position_idx < NUM_TARGET_POS) {
						set_new_target_pos(hstate_machine);
						target_changed = 1;
					}
				}
				#endif
				break;
			case TYPE_SYS_SHUTDOWN:
				land_drones = 1;
				break;
			case TYPE_TRAJECTORY:
				if (rx_data[i]->trajectory_message.drone_id == hstate_machine->id) {
					current_traj = hstate_machine->get_current_trajectory();
					uint8_t update_data = 1;
					uint32_t traj_start_time = rx_data[i]->trajectory_message.trajectory_start_time;
					
					/*if (updated) {
						if (current_traj->start_round >= traj_start_time) {
							update_data = 0;
						} else if (current_traj->calculated_by >= rx_data[i]->trajectory_message.calculated_by) {
							update_data = 0;
						}
					}*/
					// we already follow this trajectory
					if (current_traj->start_round == traj_start_time && current_traj->calculated_by == rx_data[i]->trajectory_message.calculated_by) {
						update_data = 0;
					} 
					cf_trajectory new_traj;
					if (update_data) {
						message_to_trajectory(hstate_machine, &rx_data[i]->trajectory_message, &new_traj);
						hstate_machine->current_traj = (rx_data[i]->trajectory_message);  // we save also this, because we then do not have to quantize the trajectory again-
						hstate_machine->set_current_trajectory(&new_traj);
					}
				}
				break;
			case TYPE_REQ_TRAJECTORY:
				// do nothing at the moment. First we have to analyze all the trajectories received.
				break;
			case TYPE_METADATA:
				*round_nmbr = rx_data[i]->metadata_message.round_nmbr;
				break;
			case TYPE_START_SYNC_MOVEMENT:
				hstate_machine->state = SYNC_MOVEMENT_STATE;
				break;
			// ignore others
			default:
				break;
		}
	}

	// first message we send is the current state.
	uint16_t num_messages_to_send = 1;
	float state[9];
	hstate_machine->get_cf_state(state);
	tx_data->header.type = TYPE_DRONE_STATE;
	tx_data->header.id = hstate_machine->id;
	quantize_state(&tx_data->state_message, state);
	// Change status and target if reached target.
	tx_data->state_message.status = STATUS_FLYING;
	if (CF_DIST(hstate_machine->current_target, state) < 3*0.05f) {
		num_current_targets_reached++;
		tx_data->state_message.status = STATUS_CURRENT_TARGET_REACHED;
	}

	// if all drones have reached their current target, either set a new target point of land drones
	if (num_current_targets_reached == hstate_machine->num_drones && !target_changed) {
		if (hstate_machine->target_position_idx < NUM_TARGET_POS-1) {
			hstate_machine->target_position_idx++;
			set_new_target_pos(hstate_machine);
			tx_data->state_message.status = STATUS_FLYING;
		} else {
			// now all drones have reached all targets.
			tx_data->state_message.status = STATUS_ALL_TARGETS_REACHED;
		}
	}

	#ifdef DYNAMIC_SWARM
	set_new_target_pos(hstate_machine);
	#if ROTATE_CCW
	hstate_machine->current_target_angle += 0.2f * 2 * 3.1415926535f / 16;
	#else
	hstate_machine->current_target_angle -= 0.2f * 2 * 3.1415926535f / 16;
	#endif
	#endif

	state_message_write_target_pos((state_message_t *) tx_data, hstate_machine);
	state_message_write_traj_metadata((state_message_t *) tx_data, hstate_machine);

	// if other CUs requested this trajectory, send it.
	for (uint16_t i = 0; i < size; i++) {
		if (rx_data[i]->header.type == TYPE_REQ_TRAJECTORY && rx_data[i]->trajectory_req_message.drone_id == hstate_machine->id) {
			tx_data[num_messages_to_send].trajectory_message = hstate_machine->current_traj;
			tx_data[num_messages_to_send].header.id = rx_data[i]->header.id; // send the message via the slot of the requesting cu.
			num_messages_to_send++;
		}
	}

	if (land_drones) {
		hstate_machine->state = SYS_SHUTDOWN_STATE;
	}
	return num_messages_to_send;

}

static uint16_t process_SYNC_MOVEMENT_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// iterate through array
	cf_trajectory *current_traj;
	for (uint16_t i = 0; i < size; i++) {
		switch (rx_data[i]->header.type)
		{
			case TYPE_SYNC_MOVEMENT_MESSAGE:
				// just an oscillation in x direction.
				current_traj = hstate_machine->get_current_trajectory();
				float x_state[3];
				float y_state[3];
				float z_state[3];
				hstate_machine->get_setpoint_state(x_state, y_state, z_state);
				//vTaskDelay(pdMS_TO_TICKS(40));
				z_state[0] = 1.0f + 0.2f*sinf(rx_data[i]->sync_movement_message.angle*3.1415916535f / 5);
				hstate_machine->set_state_directly(x_state, y_state, z_state);
				break;
			// ignore others
			default:
				break;
		}
	}

	uint8_t status = hstate_machine->cf_launch_status();
	float state[9];
	hstate_machine->get_cf_state(state);
	tx_data->header.type = TYPE_DRONE_STATE;
	tx_data->header.id = hstate_machine->id;
	quantize_state(&tx_data->state_message, state);
	tx_data->state_message.status = status;
	state_message_write_target_pos((state_message_t *) tx_data, hstate_machine);
	state_message_write_traj_metadata((state_message_t *) tx_data, hstate_machine);
	return 1;

}

static uint16_t process_SYS_SHUTDOWN_STATE(cf_state_machine_handle *hstate_machine, ap_message_t **rx_data, uint16_t size, ap_message_t *tx_data)
{
	// iterate through array
	uint16_t num_landed_drones = 0;
	for (uint16_t i = 0; i < size; i++) {
		switch (rx_data[i]->header.type)
		{
			case TYPE_DRONE_STATE:
				// count how many drones have launched
				if (rx_data[i]->state_message.status == STATUS_LANDED) {
					num_landed_drones++;
				}
				break;
			// ignore others
			default:
				break;
		}
	}

	uint8_t status = hstate_machine->cf_land_status();
	if (status == STATUS_FLYING) {
		hstate_machine->land_cf();
	} else if (status == STATUS_LANDED) {
		//num_landed_drones++;
	}
	float state[9];
	hstate_machine->get_cf_state(state);
	tx_data->header.type = TYPE_DRONE_STATE;
	tx_data->header.id = hstate_machine->id;
	quantize_state(&tx_data->state_message, state);
	tx_data->state_message.status = status;
	state_message_write_target_pos((state_message_t *) tx_data, hstate_machine);
	state_message_write_traj_metadata((state_message_t *) tx_data, hstate_machine);

	// go back to the AP_READY_STATE, and wait for the next launch.
	if (num_landed_drones == hstate_machine->num_drones) {
		hstate_machine->state = SYS_READY_STATE;
		// set current target to first one
		hstate_machine->target_position_idx = 0;
		set_new_target_pos(hstate_machine);
	}
	return 1;
}

void run_cf_state_machine(cf_state_machine_handle *hstate_machine)
{
	hstate_machine->state = IDLE_STATE;

	ap_message_t *rx_data[MAX_NUM_RX_MESSAGES];
	ap_message_t tx_data[MAX_NUM_TX_MESSAGES];
	uint16_t i = 0;
	while (1) {
		
		uint16_t rx_size = receive_data_from(&hstate_machine->hap_com, rx_data);
		uint16_t tx_size = 0;
		uint32_t round_nmbr = 0;
		if (i%1 == 0) {
			//DEBUG_PRINT("RX!%u\n", i);
		}
		switch (hstate_machine->state)
		{
			case IDLE_STATE:
				tx_size = process_IDLE_STATE(hstate_machine, rx_data, rx_size, tx_data);
				if (i%2 == 0) {
					//DEBUG_PRINT("IDLE_STATE!\n");
				}
				break;
			case WAIT_FOR_LAUNCH_STATE:
				tx_size = process_WAIT_FOR_LAUNCH_STATE(hstate_machine, rx_data, rx_size, tx_data);
				break;
			case VERTICAL_LAUNCH_STATE:
				tx_size = process_VERTICAL_LAUNCH_STATE(hstate_machine, rx_data, rx_size, tx_data);
				break;
			case AP_READY_STATE:
				tx_size = process_AP_READY_STATE(hstate_machine, rx_data, rx_size, tx_data);
				if (i%10 == 0) {
					//DEBUG_PRINT("AP_READY_STATE!%u\n", i);
				}
				break;
			case SYS_READY_STATE:
				if (i%10 == 0) {
					//DEBUG_PRINT("SYS_READY_STATE!\n");
				}
				tx_size = process_SYS_READY_STATE(hstate_machine, rx_data, rx_size, tx_data);
				break;
			case LAUNCH_STATE:
				if (i%10 == 0) {
					//DEBUG_PRINT("Launching!\n");
				}
				tx_size = process_LAUNCH_STATE(hstate_machine, rx_data, rx_size, tx_data);
				break;
			case SYS_RUN_STATE:
				tx_size = process_SYS_RUN_STATE(hstate_machine, rx_data, rx_size, tx_data, &round_nmbr);
				break;
			case SYNC_MOVEMENT_STATE:
				tx_size = process_SYNC_MOVEMENT_STATE(hstate_machine, rx_data, rx_size, tx_data);
				break;
			case SYS_SHUTDOWN_STATE:
				tx_size = process_SYS_SHUTDOWN_STATE(hstate_machine, rx_data, rx_size, tx_data);
				break;
			default:
				break;


		}
		i++;
		hstate_machine->round_finished(round_nmbr);
		// wait till cp collects data.
		//DEBUG_PRINT("tx!\n");
		send_data_to(&hstate_machine->hap_com, tx_data, tx_size);
		//DEBUG_PRINT("txe!\n");
	}
}
