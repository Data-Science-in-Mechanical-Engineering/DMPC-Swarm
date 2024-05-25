#include  "cu_os.h"
#include "uart.h"
#include "internal_messages.h"
#include "gpi/tools.h"
#include "gpi/platform.h"
#include "gpi/interrupts.h"
#include "gpi/clocks.h"
#include "gpi/olf.h"
#include "cp_os.h"
#include <stdint.h>

static ap_com_handle hap_com;
static uint8_t send_land_command;
static metadata_message_t message;

static void rx_wait_uart()
{
  // do not waitfor rx
}

static void tx_wait_uart()
{
  
}

static uint16_t receive_data_from_AP(ap_message_t **data)
{
  return 0;
}

static void send_data_to_AP(ap_message_t *data, uint16_t size)
{
}

static uint8_t communication_finished_callback(ap_message_t *data, uint16_t size)
{
  return 0;
} 
                
static uint16_t communication_starts_callback(ap_message_t **data)
{
  if (send_land_command) {
    message.header.type = TYPE_SYS_SHUTDOWN;
    message.header.id = 100;
    data[0] = &message;
    return 1;
  }
  return 0;
}

void run_relay_os(uint8_t id, uint8_t m_send_land_command)
{ 
  send_land_command = m_send_land_command;
  init_ap_com(&hap_com, &send_uart, &receive_uart, &rx_wait_uart, &tx_wait_uart);
  init_cp_os(&receive_data_from_AP, &send_data_to_AP, &communication_finished_callback, &communication_starts_callback, id, 0, 0);
  run();
}