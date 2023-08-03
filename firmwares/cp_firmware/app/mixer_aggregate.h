#include "wireless_control.h"
#include "mixer_config.h"

unsigned int all_flags_in_agg(uint8_t *agg);

void aggregate_merge(volatile uint8_t *agg_is_valid, uint8_t *agg_local, uint8_t *agg_rx);

void aggregate_read(uint8_t *agg, uint8_t *id, uint8_t *type, uint16_t *max_size_message);