#ifndef __MIXER_CONFIG_H__
#define __MIXER_CONFIG_H__

// mixer configuration file
// Adapt the settings to the needs of your application.

#include "gpi/platform_spec.h"		// GPI_ARCH_IS_...
#include "gpi/tools.h"				// NUM_ELEMENTS()
#include "messages.h"
#include "internal_messages.h"

// GPI_ARCH_BOARD_nRF_PCA10056
// GPI_ARCH_BOARD_TUDNES_DPP2COM
#if GPI_ARCH_IS_BOARD(nRF_PCA10056)
	#define DISABLE_BOLT 1
#endif

#define NETWORK_MANAGER_ID 234

/*****************************************************************************/
typedef struct message_assignment_t_tag
{
  uint8_t id;   // id of message slot
  uint16_t size;  // slot size in byte
  uint16_t mixer_assignment_start;  // the index in mixer, the message starts
  uint16_t mixer_assignment_end;   // the index in mixer the message ends (not including this index)
  uint16_t size_end; // the size of the piece of the message in the mixer message at index mixer_assignment_end-1
} message_assignment_t;

/* basic settings ************************************************************/

// The array contains physical node IDs and their position in the array is the logical node ID.
static const uint8_t nodes[]	= {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39};//{1, 2, 3, 20, 30, 31, 32, 33, 34};//{1, 2, 20, 21};
static const uint8_t cu_nodes[] = {20, 21, 22, 35};
static const uint8_t cf_nodes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static const uint8_t relay_nodes[] = {30, 31, 32, 33, 34, 36, 37, 38};
static const uint8_t shutdown_node = 39;

// ids of message slots
// corresponding sizes (in bytes) of message slots
static message_assignment_t message_assignment[] = {
                                                    {.id=NETWORK_MANAGER_ID, .size=sizeof(network_members_message_t)},
                                                    {.id=1, .size=sizeof(state_message_t)}, 
                                                    {.id=2, .size=sizeof(state_message_t)},
                                                    {.id=3, .size=sizeof(state_message_t)},
                                                    {.id=4, .size=sizeof(state_message_t)},
                                                    {.id=5, .size=sizeof(state_message_t)},
                                                    {.id=6, .size=sizeof(state_message_t)},
                                                    {.id=7, .size=sizeof(state_message_t)},
                                                    {.id=8, .size=sizeof(state_message_t)},
                                                    {.id=9, .size=sizeof(state_message_t)},
                                                    {.id=10, .size=sizeof(state_message_t)},
                                                    {.id=11, .size=sizeof(state_message_t)},
                                                    {.id=12, .size=sizeof(state_message_t)},
                                                    {.id=13, .size=sizeof(state_message_t)},
                                                    {.id=14, .size=sizeof(state_message_t)},
                                                    {.id=15, .size=sizeof(state_message_t)},
                                                    {.id=16, .size=sizeof(state_message_t)},
                                                    {.id=20, .size=sizeof(trajectory_message_t)},
                                                    {.id=21, .size=sizeof(trajectory_message_t)},
                                                    {.id=22, .size=sizeof(trajectory_message_t)},
                                                    {.id=100, .size=sizeof(state_message_t)},
                                                    {.id=200, .size=sizeof(target_positions_message_t)}
                                                   };

// with this, one can define messages, which are reserved for the whole system beforehand in the message_area.
// for the swarm, we reserve one for the shutdown-message and one for the target positions.
static message_assignment_t constant_message_assignment[] = {
                                                    {.id=100, .size=sizeof(state_message_t)}, 
                                                    {.id=200, .size=sizeof(target_positions_message_t)}};

/*{{.id=1, .size=sizeof(state_message_t)}, 
                                                    {.id=2, .size=sizeof(state_message_t)},
                                                    {.id=20, .size=sizeof(trajectory_message_t)},
                                                    {.id=21, .size=sizeof(trajectory_message_t)}
                                                   }; */  
//static const uint8_t nodes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};

#define MX_NUM_NODES			NUM_ELEMENTS(nodes)
#define MX_INITIATOR_ID			20 // nodes[0] // 1
#define MX_PAYLOAD_ONLY			20 // 16B state and 4B control input for logging
#define MX_PAYLOAD_SIZE			65
#define DEFAULT_MODE			0
#define ROUND_LENGTH_MS                 200

#if DEFAULT_MODE == 0
	// Entries in the plants array send probability values.
	static const uint8_t plants[] = {1, 2}; //, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};

	#define MX_ROUND_LENGTH				150 // in #slots
	#define ROUND_PERIOD				GPI_TICK_MS_TO_HYBRID2(ROUND_LENGTH_MS)
	#define MX_SLOT_LENGTH				8710 //GPI_TICK_US_TO_HYBRID2(1460) //GPI_TICK_US_TO_HYBRID2(1100)
#endif



#define NUM_PLANTS				NUM_ELEMENTS(plants)
#define MX_GENERATION_SIZE 28  // + initiator


// Possible values (Gpi_Radio_Mode):
//		IEEE_802_15_4	= 1
//		BLE_1M			= 2
//		BLE_2M			= 3
//		BLE_125k		= 4
//		BLE_500k		= 5
#define MX_PHY_MODE				3
// Values mentioned in the manual (nRF52840_PS_v1.1):
// +8dBm,  +7dBm,  +6dBm,  +5dBm,  +4dBm,  +3dBm, + 2dBm,
//  0dBm,  -4dBm,  -8dBm, -12dBm, -16dBm, -20dBm, -40dBm
#define MX_TX_PWR_DBM			-8

/*****************************************************************************/
/* special settings **********************************************************/

#define MX_WEAK_ZEROS			1
#define WEAK_RELEASE_SLOT		1
#define MX_WARMSTART_RNDS		1
#define PLANT_STATE_LOGGING		0

// When SIMULATE_MESSAGES is set to 1, the first two nodes in the plants array write all NUM_PLANTS messages.
#define SIMULATE_MESSAGES		0

// turn verbose log messages on or off
// NOTE: These additional prints might take too long when using short round intervals.
#define MX_VERBOSE_STATISTICS	1
#define MX_VERBOSE_PACKETS		0
#define MX_VERBOSE_PROFILE		0
#define WC_PROFILE_MAIN			0

#define MX_SMART_SHUTDOWN		1
// 0	no smart shutdown
// 1	no unfinished neighbor, without full-rank map(s)
// 2	no unfinished neighbor
// 3	all nodes full rank
// 4	all nodes full rank, all neighbors ACKed knowledge of this fact
// 5	all nodes full rank, all nodes ACKed knowledge of this fact
#define MX_SMART_SHUTDOWN_MODE	2


/*****************************************************************************/
/* convinience macros (dpp2com platform only) ********************************/

#if GPI_ARCH_IS_BOARD(TUDNES_DPP2COM)
	#define SET_COM_GPIO1() (NRF_P0->OUTSET = BV(26))
	#define CLR_COM_GPIO1() (NRF_P0->OUTCLR = BV(26))
	#define SET_COM_GPIO2() (NRF_P0->OUTSET = BV(28))
	#define CLR_COM_GPIO2() (NRF_P0->OUTCLR = BV(28))
        #define SET_COM_GPIOCS() (NRF_P0->OUTSET = BV(13))
        #define CLR_COM_GPIOCS() (NRF_P0->OUTCLR = BV(13))
#else
	#define SET_COM_GPIO1()
	#define CLR_COM_GPIO1()
	#define SET_COM_GPIO2()
	#define CLR_COM_GPIO2()
#endif

#endif // __MIXER_CONFIG_H__
