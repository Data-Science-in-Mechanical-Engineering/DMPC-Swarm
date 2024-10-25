/*
 * internal_messages.c
 *
 *  Created on: 04.11.2022
 *      Author: mf724021
 */
#include "FreeRTOS.h"
#include "queue.h"
#include "semphr.h"
#include "task.h"
#include "internal_messages.h"

#include <stdlib.h>
#include <stdio.h>

#include <string.h>

static uint8_t buffer[SIZE_CP_COM_BUFFER];

uint16_t message_sizes(uint8_t type)
{
  switch(type) {
    case TYPE_TRAJECTORY:
        return sizeof(trajectory_message_t);
    case TYPE_DRONE_STATE:
    	return sizeof(state_message_t);
    case TYPE_REQ_TRAJECTORY:
    	return sizeof(trajectory_req_message_t);
    case TYPE_EMTPY_MESSAGE:
        return sizeof(empty_message_t);
	case TYPE_START_SYNC_MOVEMENT:
        return sizeof(metadata_message_t);
    case TYPE_SYNC_MOVEMENT_MESSAGE:
        return sizeof(sync_movement_message_t);
    case TYPE_TARGET_POSITIONS_MESSAGE:
        return sizeof(target_positions_message_t);
    case TYPE_NETWORK_MEMBERS_MESSAGE:
        return sizeof(network_members_message_t);
    case TYPE_NETWORK_MESSAGE_AREA_REQUEST:
        return sizeof(network_area_request_message_t);
    case TYPE_NETWORK_MESSAGE_AREA_FREE:
         return sizeof(network_area_free_message_t);
    case TYPE_AP_DATA_REQ:
        return 0;
    default:
        return sizeof(metadata_message_t);
  }

}


void init_ap_com(ap_com_handle *hap_com, void (*ap_send_p)(uint8_t *, uint16_t), void (*ap_receive_p)(uint8_t *, uint16_t), void (*rx_wait_p)(), void (*tx_wait_p)())
{
	hap_com->raw_data_buffer = buffer;
	hap_com->ap_send = ap_send_p;
    hap_com->ap_receive = ap_receive_p;
	hap_com->tx_wait = tx_wait_p;
    hap_com->rx_wait = rx_wait_p;
}

uint16_t get_size(const ap_message_t *message)
{
	switch (message->header.type) {
		case TYPE_ERROR:
		case TYPE_METADATA:
		case TYPE_AP_ACK:
		case TYPE_CP_ACK:
		case TYPE_ALL_AGENTS_READY:
			return sizeof(metadata_message_t);
			break;
		default:
			return message_sizes(message->header.type);
	}
}

uint16_t raw_data_to_messages(const uint8_t *raw_data, ap_message_t **messages, uint16_t size)
{
	uint16_t raw_idx = 0;
	uint16_t messages_idx = 0;
	while (raw_idx < size) {
		messages[messages_idx] = (ap_message_t *) &raw_data[raw_idx];
		messages_idx++;
		raw_idx += get_size((ap_message_t *) &raw_data[raw_idx]);
    }
	// return length of decoded messages
	return messages_idx;
}

uint16_t messages_to_raw_data(uint8_t *raw_data, const ap_message_t *messages, uint16_t size)
{
	uint16_t raw_idx = 0;
	uint16_t messages_idx = 0;
	while (messages_idx < size) {
		uint16_t message_size = get_size(&messages[messages_idx]);
		memcpy((void *) &raw_data[raw_idx], (void *) &messages[messages_idx], message_size);
		messages_idx++;
		raw_idx += message_size;
	}
	// return length of encoded messages
	return raw_idx;
}

void send_data_to(ap_com_handle *hap_com, const ap_message_t *messages, uint16_t size)
{
	uint16_t raw_data_size = messages_to_raw_data(hap_com->raw_data_buffer, messages, size);
	// send size of to sent data
	hap_com->ap_send((uint8_t *) &raw_data_size, 2);
	hap_com->tx_wait();
	// send data
	hap_com->ap_send(hap_com->raw_data_buffer, raw_data_size);
}

uint16_t receive_data_from(ap_com_handle *hap_com, ap_message_t **messages)
{
	// receive how many bytes to expect
	uint16_t size = 0;
	hap_com->ap_receive((uint8_t *) &size, 2);
	hap_com->rx_wait();
	hap_com->ap_receive(hap_com->raw_data_buffer, size);
	return raw_data_to_messages(hap_com->raw_data_buffer, messages, size);
}



