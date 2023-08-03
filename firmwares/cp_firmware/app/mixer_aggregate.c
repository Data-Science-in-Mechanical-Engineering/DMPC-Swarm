#include "mixer_aggregate.h"

unsigned int all_flags_in_agg(uint8_t *agg)
{
  return 0;
}

static uint16_t get_max_size_message(uint8_t *agg)
{
  // bytes 3 and 4 are the maximum size of the message
  return *((uint16_t *) (agg+2));
}

void aggregate_merge(volatile uint8_t *agg_is_valid, uint8_t *agg_local, uint8_t *agg_rx) 
{
        // invalidate aggregate before modification
	*agg_is_valid = 0;

        // we can only add one agent per round. use the one with the highest message size
	if (get_max_size_message(agg_rx) > get_max_size_message(agg_rx)) {
          memcpy(agg_local, agg_rx, AGGREGATE_SIZE);
        }
        
	// activate aggregate after modification
	*agg_is_valid = 1;
  
}

void aggregate_read(uint8_t *agg, uint8_t *id, uint8_t *type, uint16_t *max_size_message)
{
 *id = agg[0];
 *type = agg[1];
 *max_size_message = get_max_size_message(agg);
}

