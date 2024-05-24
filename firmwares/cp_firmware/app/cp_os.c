#include "cp_os.h"
#include "spi.h" 
#include "mixer/mixer.h"
#include "mixer_aggregate.h"
#include "gpi/trace.h"
#include "gpi/tools.h"
#include "gpi/platform.h"
#include "gpi/interrupts.h"
#include "gpi/clocks.h"
#include "gpi/olf.h"
#include "message_layer.h"
#include "internal_messages.h"
#include "network_manager.h"

#include <string.h>

static unsigned int send_idx = -1u;
static unsigned int plant_idx = -1u;
static uint8_t node_id;
static uint32_t round;
static uint8_t agg_input[AGGREGATE_SIZE];
static Gpi_Hybrid_Tick t_ref;
static uint8_t TOS_NODE_ID;

static uint16_t (*receive_data_from_AP)(ap_message_t **);
static void (*send_data_to_AP)(ap_message_t *, uint16_t);
static uint8_t (*communication_finished_callback)(ap_message_t*, uint16_t);
static uint16_t (*communication_starts_callback)(ap_message_t**);

static ap_message_t dummy_message;
static uint8_t ap_connected;

static uint8_t is_network_manager = 0;

static uint8_t leave_network[MAX_NUM_AGENTS] = {0};

static uint8_t received_member_message_last_round = 0;

static network_members_message_t network_manager_state;
static network_members_message_t network_members_message;

void init_cp_os(uint16_t (*receive_data_from_AP_p)(ap_message_t **), 
                void (*send_data_to_AP_p)(ap_message_t *, uint16_t), 
                uint8_t (*communication_finished_callback_p)(ap_message_t*, uint16_t), 
                uint16_t (*communication_starts_callback_p)(ap_message_t**),
                uint8_t id,
                uint8_t m_ap_connnected,
                uint8_t m_is_network_manager)
{
  receive_data_from_AP = receive_data_from_AP_p;
  send_data_to_AP = send_data_to_AP_p;
  communication_finished_callback = communication_finished_callback_p;
  communication_starts_callback = communication_starts_callback_p;
  TOS_NODE_ID = id;
  ap_connected = m_ap_connnected;

  init_network_manager(&network_members_message);


  for (uint16_t i = 0; i < NUM_PLANTS; i++) {
    if (plants[i] == TOS_NODE_ID)
    {
      send_idx = i + 1;
      break;
    }
  }
  // get which agent and node we are.
  unsigned int i;
  for (uint16_t i = 0; i < NUM_PLANTS; i++) {
    if (plants[i] == TOS_NODE_ID)
    {
      plant_idx	= i;
      break;
    }
  }

  for (node_id = 0; node_id < NUM_ELEMENTS(nodes); ++node_id) {
    if (nodes[node_id] == TOS_NODE_ID)
            break;
  }
}
            

void run()
{
  ap_message_t ap_pkt;
  message_layer_init();
  if (ap_connected) {
    wait_for_AP(&ap_pkt);
    is_network_manager = ap_pkt.metadata_message.is_initiator;
  }
  //is_network_manager = TOS_NODE_ID == 20;
  init_network_manager(&network_members_message);

  if (is_network_manager) {
    init_network_manager(&network_manager_state);

    // directly add the constant message areas to the network manager
    for (uint8_t i = 0; i < NUM_ELEMENTS(constant_message_assignment); i++) {
      add_new_message(&network_manager_state, constant_message_assignment[i].id, constant_message_assignment[i].size);
    }
  }

  round = 1;
  // t_ref for first round is now (-> start as soon as possible)
  t_ref = gpi_tick_hybrid();
  //wait_for_other_agents();
  if (ap_connected) {
    // send all agents ready to AP
    ap_message_t tx_pkt;
    tx_pkt.header.type = TYPE_ALL_AGENTS_READY;
    tx_pkt.header.id = (uint8_t) TOS_NODE_ID;
    tx_pkt.metadata_message.is_initiator = ap_pkt.metadata_message.is_initiator;
    send_data_to_AP(&tx_pkt, 1);
  }
  run_normal_operation();
}

void wait_for_AP(ap_message_t *AP_pkt)
{
   ap_message_t tx_pkt;
   ap_message_t *rx_pkt;
   rx_pkt = AP_pkt;  // just to init the pointer.
   uint8_t ap_ready = 0;
   AP_pkt->header.type = TYPE_ERROR;
   while (rx_pkt->header.type != TYPE_AP_ACK) {
      NRF_P0->OUTSET = BV(25);

      // send metadata to AP
      tx_pkt.header.type = TYPE_METADATA;
      tx_pkt.header.id = (uint8_t) TOS_NODE_ID;
      tx_pkt.metadata_message.num_computing_units = NUM_ELEMENTS(cu_nodes);
      tx_pkt.metadata_message.num_drones = NUM_ELEMENTS(cf_nodes);
      tx_pkt.metadata_message.round_length_ms = ROUND_LENGTH_MS;
      send_data_to_AP(&tx_pkt, 1);
      
      // wait for ap to process data
      gpi_milli_sleep(100);

      // try to receive data from AP.
      rx_pkt->header.type = TYPE_ERROR;
      uint16_t length_received = receive_data_from_AP(&rx_pkt);
      /*while(1){
      gpi_milli_sleep(500);
      NRF_P0->OUTCLR = BV(25);
      gpi_milli_sleep(1000);
      NRF_P0->OUTSET = BV(25);}*/
      // if AP sent more than one message something is wrong.
      assert(length_received <= 1);
      gpi_milli_sleep(500);
      NRF_P0->OUTCLR = BV(25);
      gpi_milli_sleep(50);
   }
   memcpy(AP_pkt, rx_pkt, sizeof(metadata_message_t));
}

void run_rounds(uint8_t (*communication_finished_callback)(ap_message_t*, uint16_t), uint16_t (*communication_starts_callback)(ap_message_t**))
{
  init_message_t init_pkt = {.round = 0};
  // init buffer, which saves messages, which are received
  ap_message_t mixer_messages_received[NUM_ELEMENTS(message_assignment) + 1]; //+1, because the las entry is the init_pkt
  // if 1, message is valid, if 0, message is not valid (was not received)
  uint8_t mixer_messages_received_valid[MX_GENERATION_SIZE-1];
  for (; 1; round++) {
    // init mixer
    mixer_init(node_id);
    mixer_set_weak_release_slot(WEAK_RELEASE_SLOT);
    mixer_set_weak_return_msg((void*)-1);
    // init aggregate callback (not important if we do not use aggregates, which we currently not do)
    mixer_init_agg(&aggregate_merge);
    // reset aggregate
    memset(agg_input, 0, AGGREGATE_SIZE);

    // Initiator sends initiator packet. Currently it only holds the round number
    //if (MX_INITIATOR_ID == TOS_NODE_ID)
    if (is_network_manager) {
      init_pkt.round = round;
                           
      // NOTE: we specified that the control packet uses index 0 and data packets use
      // indexes > 0. 
      mixer_write(0, &init_pkt, sizeof(init_message_t));
    }
    SET_COM_GPIO1();
    
    // wait before calling callback to give application processor enough time for computation
    while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), READ_AND_ARM_OFFSET(t_ref, ROUND_PERIOD)) < 0);
    CLR_COM_GPIO1();

    // e.g. read data from application processor (or do something else)
    ap_message_t *tx_messages[NUM_ELEMENTS(message_assignment)];
    uint16_t size_tx_messages = communication_starts_callback(tx_messages);

    // check, if the area should be freed.
    for (uint16_t tx_message_idx = 0; tx_message_idx < size_tx_messages; tx_message_idx++) {
        if (tx_messages[tx_message_idx]->header.type == TYPE_NETWORK_MESSAGE_AREA_FREE) {
          uint8_t idx = get_message_area_idx(&network_members_message, tx_messages[tx_message_idx]->header.id);
          leave_network[idx] = 1;
        }
    }

    
    // write into mixer
    for (uint16_t tx_message_idx = 0; tx_message_idx < size_tx_messages; tx_message_idx++) {

      // when the agent does not want to send anything, it sends a TYPE_DUMMY
      if (tx_messages[tx_message_idx]->header.type != TYPE_DUMMY 
          && tx_messages[tx_message_idx]->header.type != TYPE_NETWORK_MESSAGE_AREA_REQUEST) {
        // the id of the message in the message layer is written in the header.
        uint8_t idx = get_message_area_idx(&network_members_message, tx_messages[tx_message_idx]->header.id);
        
        // only if the area should not be freed or if it should be freed and we recenved the member message last round,
        // then send (otherwise we do not know if the message area was freed already)
        //if (idx == 255) {
          //while(1) {}
        //}
        
        if (received_member_message_last_round || !leave_network[idx]) {
          if (idx != 255) {
            message_layer_set_message(idx, 
                (uint8_t *) tx_messages[tx_message_idx]);
          }
        }
      }

      // write into aggregate that the ap wants to reserve a new area in the message layer.
      if (tx_messages[tx_message_idx]->header.type == TYPE_NETWORK_MESSAGE_AREA_REQUEST) {
        // it takes two round for an AP to get notified by the network manager, that its request was succesfull.
        // It thus ma send two request. Thats why we check if the requested id was already in the message layer.
        if (!id_already_in_message_layer(&network_manager_state, tx_messages[tx_message_idx]->network_area_request_message.id)) {
          agg_input[0] = tx_messages[tx_message_idx]->network_area_request_message.id;
          agg_input[1] = tx_messages[tx_message_idx]->network_area_request_message.type;
          agg_set_max_size_message(agg_input, tx_messages[tx_message_idx]->network_area_request_message.max_size_message);
        }
      }
    }

    mixer_write_agg(agg_input);

    if (is_network_manager) {
      // index 0 is always the network manager message.
      message_layer_set_message(0, (uint8_t *) &network_manager_state);
    }

    // arm mixer
    // start first round with infinite scan
    // -> nodes join next available round, does not require simultaneous boot-up
    // mixer_arm(((MX_INITIATOR_ID == TOS_NODE_ID) ? MX_ARM_INITIATOR : 0) | ((1 == round) ? MX_ARM_INFINITE_SCAN : 0));
    mixer_arm(((is_network_manager) ? MX_ARM_INITIATOR : 0) | ((1 == round) ? MX_ARM_INFINITE_SCAN : 0));

    // poll such that mixer round starts at the correct time.
    // delay initiator a bit
    // -> increase probability that all nodes are ready when initiator starts the round
    // -> avoid problems in view of limited t_ref accuracy
    SET_COM_GPIO1();
    // if (MX_INITIATOR_ID == TOS_NODE_ID)
    if (is_network_manager) {
      while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), MIXER_OFFSET(t_ref, ROUND_PERIOD) + MIXER_INITIATOR_DELAY) < 0);
    }
    else {
      while (gpi_tick_compare_hybrid(gpi_tick_hybrid(), MIXER_OFFSET(t_ref, ROUND_PERIOD)) < 0);
    }
    CLR_COM_GPIO1();      
    NRF_P0->OUTCLR = BV(25);
    // ATTENTION: don't delay after the polling loop (-> print before)
    t_ref = mixer_start();
    NRF_P0->OUTSET = BV(25);

    received_member_message_last_round = 0;
      
    // Just for Debug
    SET_COM_GPIO1();

    // read received data to AP
    uint32_t msgs_not_decoded = 0;
    uint32_t msgs_weak = 0;
    uint32_t control_msg_decoded = 0;
    uint16_t messages_received_idx = 0;
    for (uint16_t i = 0; i < NUM_ELEMENTS(message_assignment); i++) {
      // write data in array, when message was received
      messages_received_idx += message_layer_get_message(i, (uint8_t *) &mixer_messages_received[messages_received_idx]);
    }
    // synchronize to the initiator node
    init_message_t init_message;
    uint8_t succ = read_message_from_mixer(0, (uint8_t *) &init_message, sizeof(init_message_t));
    if (succ) {
      if (1 == round) {
        round = init_message.round;
      // resynchronize when round number does not match
      } else if (init_message.round != round) {
        round = 0;	// increments to 1 with next round loop iteration
      }
    }

    // write round in metadata message, which will be sent to AP, such that AP knows what time it is.
    mixer_messages_received[messages_received_idx].metadata_message.header.type = TYPE_METADATA;
    mixer_messages_received[messages_received_idx].metadata_message.header.id = TOS_NODE_ID;
    mixer_messages_received[messages_received_idx].metadata_message.round_nmbr = round;
    messages_received_idx += 1;
    // process received data (e.g. send it to AP) and finish, if the callback says so.
    if (communication_finished_callback(mixer_messages_received, messages_received_idx)) {
      break;
    }

    // now process first received message if the first received messge is from the network manager (we have received it), update our local one.
    // the AP does not receive our local one, because for its algrorithms it might be necessary to know if it has received the current
    // state of the network manager. For the CP this is not important, because as long as we are in the network, the message areas,
    // we write to, will always be the network areas, we reserved, even if we do not receive the network managers message.
    if (mixer_messages_received[0].header.type == TYPE_NETWORK_MEMBERS_MESSAGE) {
      received_member_message_last_round = 1;
      memcpy(&network_members_message, &mixer_messages_received[0], sizeof(network_members_message_t));
    } else {
      // if we have not received it, then countdown -1
      if (network_members_message.id_new_network_manager != 0) {
        network_members_message.manager_wants_to_leave_network_in -= 1;
      }
    }

    // if the countdown reached 0 and we are the new network manager, set ourself as networkm manager or do not set ourself
    if (network_members_message.manager_wants_to_leave_network_in == 0 && network_members_message.id_new_network_manager != 0) {
      if (network_members_message.id_new_network_manager==TOS_NODE_ID) {
        is_network_manager = 1;
      
        init_network_manager(&network_manager_state);
        memcpy(&network_manager_state, &network_members_message, sizeof(network_members_message_t));

        // we are now the network manager, stop the request for it.
        network_manager_state.id_new_network_manager = 0;
      } else {
        // we are not network manager anymore.
        if (is_network_manager) {
          is_network_manager = 0;
        }
      }
    }

    // check if an agent wants to leave the network
    if (is_network_manager) {
      for (uint8_t i = 0; i < messages_received_idx; i++) {
        // if this node is the agents that wants to leave, start countdown and assign a new agent
        if (mixer_messages_received[i].header.id == TOS_NODE_ID 
            && mixer_messages_received[i].header.type == TYPE_NETWORK_MESSAGE_AREA_FREE
            && !network_manager_state.id_new_network_manager) {
          for (uint8_t i = 0; i < MAX_NUM_AGENTS; i++) {
            // search for another CU.
            if (network_manager_state.types[i] == 0 && network_manager_state.ids[i] != TOS_NODE_ID) {
              network_manager_state.id_new_network_manager = network_manager_state.ids[i];
              break;
            } 
          }
          network_manager_state.manager_wants_to_leave_network_in = 5; 
        } else {
          // only if the network manager does not want to leave the network, remove agents from the network.
          // Otherwise, the new manager might not notice the changes in the network, if it misses the message.
          if (!network_manager_state.id_new_network_manager && mixer_messages_received[i].header.type == TYPE_NETWORK_MESSAGE_AREA_FREE) {
            remove_agent(&network_manager_state, mixer_messages_received[i].header.id);
          }
        }
      }

      if (network_manager_state.id_new_network_manager) {
        network_manager_state.manager_wants_to_leave_network_in -= 1;
      }

    }

    // read and process aggregate
    if (is_network_manager) {
      uint8_t *agg_rx = mixer_read_agg();
      uint8_t id = 0;
      uint8_t type = 0;
      uint16_t max_size_message = 0;
      aggregate_read(agg_rx, &id, &type, &max_size_message);
      // if id is not 0, then a new message is registered
      if (id != 0 && !network_manager_state.id_new_network_manager) {
        // if the type is 255, then an already existing agent wants to reserve an other message area.
        if (type != 255) {
          add_new_agent(&network_manager_state, id, type, max_size_message);
        } else {
          add_new_message(&network_manager_state, id, max_size_message);
        }
      }
    }
  }
}

static uint16_t wait_for_agents_com_starts_callback(ap_message_t **message)
{
  dummy_message.header.type = TYPE_CP_ACK;
  dummy_message.header.id = TOS_NODE_ID;
  message[0] = &dummy_message;
  //NRF_P0->OUTSET = BV(25);
  return 1;
}

static uint8_t wait_for_agents_com_finished_callback(ap_message_t *received_messages, uint16_t size)
{
  /*uint16_t num_messages_received = 0;
  for (uint16_t i = 0; i<NUM_ELEMENTS(message_assignment)+1; i++) {
    if (messages_received_valid[i] == 1) {
      num_messages_received++;
    }
  }*/
  if (size == ((uint16_t) NUM_ELEMENTS(message_assignment))+1) {
    return 1;
  }
  return 0;
}

void wait_for_other_agents()
{   
  run_rounds(&wait_for_agents_com_finished_callback, &wait_for_agents_com_starts_callback);
}


/**
 * run in normal operation
 */
void run_normal_operation()
{
  run_rounds(communication_finished_callback, communication_starts_callback);
}