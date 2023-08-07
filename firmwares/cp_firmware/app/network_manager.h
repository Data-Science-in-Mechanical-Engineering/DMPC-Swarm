#ifndef INC_NETWORK_MANAGER_H
#define INC_NETWORK_MANAGER_H

#include <stdint.h>
#include "internal_messages.h"

void init_network_manager(network_members_message_t *state);

uint8_t add_new_message(network_members_message_t *state, uint8_t message_id, uint16_t max_message_size);

uint8_t id_already_in_message_layer(network_members_message_t *state, uint8_t id);

/**
 * add a new agent to the network. Every agent gets its own message area. 
 * if the agent wants to reserve more than one message area, it has to use add_new_message
 */
uint8_t add_new_agent(network_members_message_t *state, uint8_t id, uint8_t type, uint16_t max_message_size);

void remove_agent(network_members_message_t *state, uint8_t agent_id);

uint8_t get_message_area_idx(network_members_message_t *state, uint8_t message_id);

#endif  /* INC_NETWORK_MANAGER_H */