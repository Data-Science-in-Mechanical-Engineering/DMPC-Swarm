#include "network_manager.h"
#include "mixer_config.h"

void init_network_manager(network_members_message_t *state)
{
  for (uint8_t i = 0; i < MAX_NUM_AGENTS; i++) {
    state->message_layer_area_agent_id[i] = 0;
    state->ids[i] = 0;
    state->types[i] = 255;
  }

  // the first field is the network area is reserved for the network manager
  state->message_layer_area_agent_id[0] = NETWORK_MANAGER_ID;
  state->header.id = NETWORK_MANAGER_ID;
  state->header.type = TYPE_NETWORK_MEMBERS_MESSAGE;
}

uint8_t id_already_in_message_layer(network_members_message_t *state, uint8_t id) {
  for (uint8_t i = 0; i < MAX_NUM_AGENTS; i++) {
    if (id == state->message_layer_area_agent_id[i]) {
      return 1;
    }
  }
  return 0;
}

uint8_t add_new_message(network_members_message_t *state, uint8_t message_id, uint16_t max_message_size)
{
  // it takes two round for an AP to get notified by the network manager, that its request was succesfull.
  // It thus ma send two request. Thats why we check if the requested id was already in the message layer.
  if (id_already_in_message_layer(state, message_id)) {
    return 1;
  }

  // search for a message layer_area, which is not used and has a size bigger than max_message_size
  uint8_t best_area = 255;
  for (uint8_t i = 0; i < NUM_ELEMENTS(message_assignment); i++) {
    // area not not used
    if (state->message_layer_area_agent_id[i] == 0) {
      if (message_assignment[i].size >= max_message_size) {
        if (best_area == 255) {
          best_area = i;
        } else {
          if (message_assignment[best_area].size > message_assignment[i].size) {
            best_area = i;
          }
        }
      }
    }
  }

  if (best_area != 255) {
    state->message_layer_area_agent_id[best_area] = message_id;
    return 0;
  }

  // could not find a suitable area.
  return 1;
}

uint8_t add_new_agent(network_members_message_t *state, uint8_t agent_id, uint8_t type, uint16_t max_message_size)
{
  uint8_t nsucc = add_new_message(state, agent_id, max_message_size);
  if (nsucc) {
    return 1;
  }

  for (uint8_t i = 0; i < MAX_NUM_AGENTS; i++) {
    if (state->ids[i] == 0) {
      state->ids[i] = agent_id;
      state->types[i] = type;
      return 0;
    }
  }
  return 1;

}

void remove_agent(network_members_message_t *state, uint8_t agent_id) {
  for (uint8_t i = 0; i < MAX_NUM_AGENTS; i++) {
    if (state->message_layer_area_agent_id[i] == agent_id) {
      state->message_layer_area_agent_id[i] == 0;
    }
    if (state->ids[i] == agent_id) {
      state->ids[i] = 0;
      state->types[i] = 255;
    }
  }
}

uint8_t get_message_area_idx(network_members_message_t *state, uint8_t message_id)
{
  for (uint8_t i = 0; i < MAX_NUM_AGENTS; i++) {
    if (state->message_layer_area_agent_id[i] == message_id) {
      return i;
    }
  }
  return 255;
}