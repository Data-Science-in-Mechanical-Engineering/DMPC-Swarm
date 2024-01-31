#include  "cf_os.h"
#include "spi.h"
#include "internal_messages.h"
#include <stdint.h>

static ap_com_handle hap_com;

static void wait_ap_spi()
{
  gpi_micro_sleep(100);
}

static uint16_t receive_data_from_AP(ap_message_t **data)
{
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
  // just forward all data which was received from the cf
  // because CP is SPI, master, there is no TYPE_AP_DATA_REQ packet needed to send
  printf("Start rx\r\n");
  uint16_t size = receive_data_from_AP(data);
  printf("End rx %u\r\n", size);
  return size;
}

void run_cf_os(uint8_t id)
{
  init_ap_com(&hap_com, &spi_tx, &spi_rx, &wait_ap_spi, &wait_ap_spi);
  init_cp_os(&receive_data_from_AP, &send_data_to_AP, &communication_finished_callback, &communication_starts_callback, id, 1, 0);
  run();
}