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

static void rx_wait_uart()
{
  // do not waitfor rx
}

static void tx_wait_uart()
{
  gpi_micro_sleep(10);
}

static uint16_t receive_data_from_AP(ap_message_t **data)
{
  // send TYPE_AP_DATA_REQ to notify CU, that it should start UART communication.
  ap_message_t req;
  req.header.type = TYPE_AP_DATA_REQ;
  send_data_to(&hap_com, &req, 1);
  return receive_data_from(&hap_com, data);
}

static void send_data_to_AP(ap_message_t *data, uint16_t size)
{
  send_data_to(&hap_com, data, size);
}

static uint8_t communication_finished_callback(ap_message_t *data, uint16_t size)
{
  // send all received messages to AP.
  send_data_to_AP(data, size);
  return 0;
} 
                
static uint16_t communication_starts_callback(ap_message_t **data)
{
  // just forward all data which was received from the CU
  uint16_t size = receive_data_from_AP(data);
  return size;
}

void run_cu_os(uint8_t id)
{ 
  init_ap_com(&hap_com, &send_uart, &receive_uart, &rx_wait_uart, &tx_wait_uart);
  init_cp_os(&receive_data_from_AP, &send_data_to_AP, &communication_finished_callback, &communication_starts_callback, id, 1);

  run();
}