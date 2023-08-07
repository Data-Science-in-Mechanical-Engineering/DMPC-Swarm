/*
 * internal_messages.h
 *
 * messages, which are send via spi and mixer. Please define your messages in messages.h. via typedef mixer_data_t your_message_t
 * in order to get the length of the messages, define a makro MESSAGES_SIZES(type), which returns the size of the message for its type.
 * Define the type of the message starting with 0 counting up.
 *  Created on: Mar 18, 2022
 *      Author: mf724021
 */

#ifndef INC_INTERNAL_MESSAGES_H_
#define INC_INTERNAL_MESSAGES_H_

#include <stdint.h>

#define TYPE_ERROR 0  // when something went wrong and the following data should not been used (e.g. CP did not receive any new data from AP)
                      // can be used to check if AP is online and ready
#define TYPE_METADATA 1  // message contains metadata about itself (sent by CP, when CP is booting up)
#define TYPE_AP_ACK 2 // AP has acknowledged that CP is online and AP is ready.
#define TYPE_CP_ACK 3 // CP acknowledged that AP is ready. AP should now wait for CP to find connection to other CPs
#define TYPE_ALL_AGENTS_READY 4  // all agents are ready
#define TYPE_AP_DATA_REQ 5      // request data from AP this message is send right before the CP wants to receive data from the AP.

#define TRANSFORM_TYPE_BACK(type) (type - TYPE_AP_DATA_REQ - 1)  // the type, which is received by mixer or spi
                                                                  // transformed to the type defined in messages.h
#define TRANSFORM_TYPE(type) (type + TYPE_AP_DATA_REQ + 1)       // the type defined in messages.h,
                                                                  // transformed to the type, which should be sent via spi and mixer
#define TYPE_DUMMY 255                                            // the AP sends this message (metadata), when the CP should ignore this message (we need this, because otherwise the UART would not work)

#define MAX_NUM_TX_MESSAGES 20
#define MAX_NUM_RX_MESSAGES 20

typedef struct __attribute__((packed)) message_t_tag
{
        uint8_t type;
        uint8_t id; //ID of Agent, which sends the message (or the slot of the mesage)
} message_t;

/**
 * struct sent in order to share state of whole system with AP
 */
typedef struct __attribute__((packed)) metadata_message_t_tag
{
        message_t header;
        uint8_t num_computing_units;
        uint8_t num_drones;
        uint16_t round_length_ms;   // currently unused, TODO change it
        uint8_t own_id;
        uint16_t round_nmbr;
} metadata_message_t;

typedef struct __attribute__((packed)) init_message_t_tag
{
        uint32_t round;
} init_message_t;

/***********user defined messages***************/
// ATTENTION do not forget to overwrite uint16_t message_sizes(uint8_t type) in internal_messages.c

#define LENGTH_TRAJECTORY 15

#define LENGTH_STATE 9

#define TYPE_LAUNCH_DRONES TRANSFORM_TYPE(0)
#define TYPE_TRAJECTORY TRANSFORM_TYPE(1)
#define TYPE_DRONE_STATE TRANSFORM_TYPE(2)
#define TYPE_REQ_TRAJECTORY TRANSFORM_TYPE(3)
#define TYPE_SYS_SHUTDOWN TRANSFORM_TYPE(4)
#define TYPE_EMTPY_MESSAGE TRANSFORM_TYPE(5)
#define TYPE_START_SYNC_MOVEMENT TRANSFORM_TYPE(6)   // when this message is sent, the drones make a synchronized wave movement
#define TYPE_SYNC_MOVEMENT_MESSAGE TRANSFORM_TYPE(7)
#define TYPE_TARGET_POSITIONS_MESSAGE TRANSFORM_TYPE(8)

#define MESSAGES_SIZES(type) message_sizes(type)

#define MAXIMUM_NUMBER_MESSAGES 100 // maximum number expected for one AP-CP communication round

#define MAX_NUM_DRONES 10

typedef struct __attribute__((packed)) trajectory_message_t_tag
{
	message_t header;
	uint16_t trajectory[LENGTH_TRAJECTORY*3];
	uint16_t init_state[LENGTH_STATE];  // first position, then velocity then acc
	uint16_t trajectory_start_time;    // trajectory start time in rounds
	uint8_t drone_id; // to which drone the trajectory belongs
	uint8_t calculated_by;
    uint8_t prios[MAX_NUM_DRONES];
} trajectory_message_t;

typedef struct __attribute__((packed)) state_message_t_tag
{
	message_t header;
	uint16_t state[LENGTH_STATE];
	uint8_t status;
	uint16_t current_target[3];
	uint8_t target_position_idx;
    uint16_t trajectory_start_time;    // trajectory start time in rounds
	uint8_t drone_id; // to which drone the trajectory belongs
	uint8_t calculated_by;
} state_message_t;

typedef struct __attribute__((packed)) trajectory_req_message_t_tag
{
    message_t header;
	uint8_t drone_id;
	uint8_t cu_id;
} trajectory_req_message_t;

typedef struct __attribute__((packed)) empty_message_t_tag
{
    message_t header;
	uint8_t cu_id;
	uint8_t prios[MAX_NUM_DRONES];
} empty_message_t;

typedef struct __attribute__((packed)) sync_movement_message_t_tag
{
        message_t header;
        uint16_t angle;
} sync_movement_message_t;

typedef struct __attribute__((packed)) target_positions_message_t_tag
{
        message_t header;
		uint8_t ids[MAX_NUM_DRONES];
        uint16_t target_positions[3 * MAX_NUM_DRONES];
} target_positions_message_t;


// write all possible messages here. This allows us to quickly transform the data from bytes to usefull structs.
typedef union ap_message_t_tag
{
	message_t header;
	metadata_message_t metadata_message;
	state_message_t state_message;
	trajectory_message_t trajectory_message;
	trajectory_req_message_t trajectory_req_message;
    empty_message_t empty_message;
	sync_movement_message_t sync_movement_message;
} ap_message_t;


typedef struct ap_com_handle_tag
{
	uint8_t *raw_data_buffer;
	uint8_t *dummy;

	void (*ap_send)(uint8_t *, uint16_t);
	void (*ap_receive)(uint8_t *, uint16_t);
	void (*rx_wait)();
    void (*tx_wait)();
} ap_com_handle;

void init_ap_com(ap_com_handle *hap_com, void (*ap_send_p)(uint8_t *, uint16_t), void (*ap_receive_p)(uint8_t *, uint16_t), void (*rx_wait_p)(), void (*tx_wait_p)());
uint16_t get_size(const ap_message_t *message);

uint16_t raw_data_to_messages(const uint8_t *raw_data, ap_message_t **messages, uint16_t size);

uint16_t messages_to_raw_data(uint8_t *raw_data, const ap_message_t *messages, uint16_t size);

void send_data_to(ap_com_handle *hap_com, const ap_message_t *messages, uint16_t size);

uint16_t receive_data_from(ap_com_handle *hap_com, ap_message_t **messages);

#endif /* INC_MESSAGES_H_ */

