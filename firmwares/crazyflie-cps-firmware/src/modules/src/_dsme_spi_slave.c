/**
 * @brief: SPI Slave Task to receive massages from cp and set the trajectory of the Drone.
 *         The SPI was original implemented as Master for decks (from crazyflie) in deck_spi.c.
 *         So first thing to do here is disabling the SPI and DMA, and then re-init them.
 *
 *         The state maschine for dealing with the massage is implemted recording the design of
 *         Alexander Gräfe. please ask him for more information.
 *
 * Author : Shengsi Xu
 * Date : 22.06.2022
 * Version : 0.1(Test version)
 */

// ST Include
#include "stm32f4xx.h"

// FreeRTOS Includes
#include "FreeRTOS.h"
#include "queue.h"
#include "semphr.h"
#include "task.h"

// Crazyflie Includes
#include "static_mem.h"
#include "config.h"
#include "cfassert.h"
#include "system.h"
#include "nvicconf.h"

#include <stdbool.h>
#include <string.h>

#include "_dsme_spi_slave.h"

#include "internal_messages.h"
#include "_dsme_trajectory_interpolation.h"
#include "cps_config.h"

#include "debug.h"

/******************** SPI DMA define ********************/
#define SPI_SLAVE SPI1
#define SPI_SLAVE_CLK RCC_APB2Periph_SPI1
#define SPI_SLAVE_CLK_INIT RCC_APB2PeriphClockCmd
#define SPI_SLAVE_IRQn SPI1_IRQn
#define SPI_SLAVE_GPIO_PORT GPIOA

#define SPI_SLAVE_DMA_IRQ_PRIO (NVIC_HIGH_PRI)
#define SPI_SLAVE_DMA DMA2
#define SPI_SLAVE_DMA_CLK RCC_AHB1Periph_DMA2
#define SPI_SLAVE_DMA_CLK_INIT RCC_AHB1PeriphClockCmd

#define SPI_SLAVE_TX_DMA_STREAM DMA2_Stream5
#define SPI_SLAVE_TX_DMA_IRQ DMA2_Stream5_IRQn
#define SPI_SLAVE_TX_DMA_IRQHandler DMA2_Stream5_IRQHandler
#define SPI_SLAVE_TX_DMA_CHANNEL DMA_Channel_3
#define SPI_SLAVE_TX_DMA_FLAG_TCIF DMA_FLAG_TCIF5

#define SPI_SLAVE_RX_DMA_STREAM DMA2_Stream0
#define SPI_SLAVE_RX_DMA_IRQ DMA2_Stream0_IRQn
#define SPI_SLAVE_RX_DMA_IRQHandler DMA2_Stream0_IRQHandler
#define SPI_SLAVE_RX_DMA_CHANNEL DMA_Channel_3
#define SPI_SLAVE_RX_DMA_FLAG_TCIF DMA_FLAG_TCIF0

#define SPI_SLAVE_SCK_PIN GPIO_Pin_5
#define SPI_SLAVE_SCK_GPIO_PORT GPIOA
#define SPI_SLAVE_SCK_GPIO_CLK RCC_AHB1Periph_GPIOA
#define SPI_SLAVE_SCK_SOURCE GPIO_PinSource5
#define SPI_SLAVE_SCK_AF GPIO_AF_SPI1

#define SPI_SLAVE_MISO_PIN GPIO_Pin_6
#define SPI_SLAVE_MISO_GPIO_PORT GPIOA
#define SPI_SLAVE_MISO_GPIO_CLK RCC_AHB1Periph_GPIOA
#define SPI_SLAVE_MISO_SOURCE GPIO_PinSource6
#define SPI_SLAVE_MISO_AF GPIO_AF_SPI1

#define SPI_SLAVE_MOSI_PIN GPIO_Pin_7
#define SPI_SLAVE_MOSI_GPIO_PORT GPIOA
#define SPI_SLAVE_MOSI_GPIO_CLK RCC_AHB1Periph_GPIOA
#define SPI_SLAVE_MOSI_SOURCE GPIO_PinSource7
#define SPI_SLAVE_MOSI_AF GPIO_AF_SPI1

/******************** Data Length Define ********************/
// Predefined Data Length in _dsme_message.h
#define BUF_IN_LEN 1024
// TODO: ACK Daten Länge festzustellen.
#define BUF_OUT_LEN BUF_IN_LEN

/******************** Privat Var ********************/
static SemaphoreHandle_t txComplete;
static SemaphoreHandle_t rxComplete;

// static uint8_t spi_slave_dma_buffer_in_0[BUF_IN_LEN];
// static uint8_t spi_slave_dma_buffer_in_1[BUF_IN_LEN];
static uint8_t spi_slave_dma_buffer_in_0[BUF_OUT_LEN];
static uint8_t spi_slave_dma_buffer_in_1[BUF_OUT_LEN];
// turn num to a buf array name
#define DMA_IN_BUF(num) \
    (num == 0 ? spi_slave_dma_buffer_in_##0 : spi_slave_dma_buffer_in_##1)
// TODO: muss noch angepsaat werden
static uint8_t spi_slave_dma_buffer_out[1024];

static bool isInit = false;

static bool mocapReady = false;

/******************** Privat Fun ********************/
static void spiGpioInit();
static void spiSlaveInit();
static void spiDMAInit();
static void spiSlaveBegin();

static void spiSlaveTask(void *param);

// Static Mem alloc. please read crazyflie documents for details.
// TODO: stack size festzustellen. Jetzt ist 3*150 Bytes. Das ist abh. von buffer_out size.
STATIC_MEM_TASK_ALLOC(spiSlaveTask, SPI_SLAVE_TASK_STACKSIZE);

/******************** Privat Fun implement ********************/
static void spiSlaveBegin()
{
    // Disable all used peripherals: SPI and DMA
    SPI_Cmd(SPI_SLAVE, DISABLE);
    SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Tx, DISABLE);
    SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Rx, DISABLE);
    DMA_Cmd(SPI_SLAVE_RX_DMA_STREAM, DISABLE);
    DMA_Cmd(SPI_SLAVE_TX_DMA_STREAM, DISABLE);

    txComplete = xSemaphoreCreateBinary();
    rxComplete = xSemaphoreCreateBinary();

    spiGpioInit();
    // SPI DMA Initialization
    spiDMAInit();
    // SPI configuration
    spiSlaveInit();

    // DMA transfer complet Interrupt (TC) enabled
    DMA_ITConfig(SPI_SLAVE_TX_DMA_STREAM, DMA_IT_TC, ENABLE);
    DMA_ITConfig(SPI_SLAVE_RX_DMA_STREAM, DMA_IT_TC, ENABLE);

    // Clear DMA Flags
    DMA_ClearFlag(SPI_SLAVE_TX_DMA_STREAM, DMA_FLAG_FEIF5 | DMA_FLAG_DMEIF5 | DMA_FLAG_TEIF5 | DMA_FLAG_HTIF5 | DMA_FLAG_TCIF5);
    DMA_ClearFlag(SPI_SLAVE_RX_DMA_STREAM, DMA_FLAG_FEIF0 | DMA_FLAG_DMEIF0 | DMA_FLAG_TEIF0 | DMA_FLAG_HTIF0 | DMA_FLAG_TCIF0);

    // Re-Enable peripherals: SPI
    SPI_Cmd(SPI_SLAVE, ENABLE);
}

static void spiGpioInit()
{
    GPIO_InitTypeDef GPIO_InitStructure;

    // Enable GPIO Clocks
    RCC_AHB1PeriphClockCmd(SPI_SLAVE_SCK_GPIO_CLK | SPI_SLAVE_MISO_GPIO_CLK |
                               SPI_SLAVE_MOSI_GPIO_CLK,
                           ENABLE);

    //GPIO_DeInit(SPI_SLAVE_GPIO_PORT);
    // SPI pins configuration
    // Connect SPI pins to AF5
    GPIO_PinAFConfig(SPI_SLAVE_SCK_GPIO_PORT, SPI_SLAVE_SCK_SOURCE, SPI_SLAVE_SCK_AF);
    GPIO_PinAFConfig(SPI_SLAVE_MISO_GPIO_PORT, SPI_SLAVE_MISO_SOURCE, SPI_SLAVE_MISO_AF);
    GPIO_PinAFConfig(SPI_SLAVE_MOSI_GPIO_PORT, SPI_SLAVE_MOSI_SOURCE, SPI_SLAVE_MOSI_AF);

    GPIO_InitStructure.GPIO_Mode = GPIO_Mode_AF;
    GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
    GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;
    GPIO_InitStructure.GPIO_PuPd = GPIO_PuPd_DOWN;

    // SPI SCK pin configuration
    GPIO_InitStructure.GPIO_Pin = SPI_SLAVE_SCK_PIN;
    GPIO_Init(SPI_SLAVE_SCK_GPIO_PORT, &GPIO_InitStructure);

    // SPI MOSI pin configuration
    GPIO_InitStructure.GPIO_Pin = SPI_SLAVE_MOSI_PIN;
    GPIO_Init(SPI_SLAVE_MOSI_GPIO_PORT, &GPIO_InitStructure);

    // SPI MISO pin configuration
    GPIO_InitStructure.GPIO_Pin = SPI_SLAVE_MISO_PIN;
    GPIO_Init(SPI_SLAVE_MISO_GPIO_PORT, &GPIO_InitStructure);
}

static void spiDMAInit()
{
    // Check if the Buffers are valid for STM32F4 DMA
    ASSERT_DMA_SAFE(&spi_slave_dma_buffer_in_0);
    ASSERT_DMA_SAFE(&spi_slave_dma_buffer_in_1);
    ASSERT_DMA_SAFE(&spi_slave_dma_buffer_out);

    DMA_InitTypeDef DMA_InitStructure;
    NVIC_InitTypeDef NVIC_InitStructure;

    // Enable DMA Clocks
    SPI_SLAVE_DMA_CLK_INIT(SPI_SLAVE_DMA_CLK, ENABLE);

    DMA_DeInit(SPI_SLAVE_RX_DMA_STREAM);
    DMA_DeInit(SPI_SLAVE_TX_DMA_STREAM);

    // Configure DMA Initialization Structure
    DMA_InitStructure.DMA_FIFOMode = DMA_FIFOMode_Disable;
    DMA_InitStructure.DMA_FIFOThreshold = DMA_FIFOThreshold_1QuarterFull;
    DMA_InitStructure.DMA_MemoryBurst = DMA_MemoryBurst_Single;
    DMA_InitStructure.DMA_MemoryDataSize = DMA_MemoryDataSize_Byte;
    DMA_InitStructure.DMA_MemoryInc = DMA_MemoryInc_Enable;

    DMA_InitStructure.DMA_PeripheralBaseAddr = (uint32_t)(&(SPI_SLAVE->DR));
    DMA_InitStructure.DMA_PeripheralBurst = DMA_PeripheralBurst_Single;
    DMA_InitStructure.DMA_PeripheralDataSize = DMA_PeripheralDataSize_Byte;
    DMA_InitStructure.DMA_PeripheralInc = DMA_PeripheralInc_Disable;
    DMA_InitStructure.DMA_Priority = DMA_Priority_High;

    // Configure TX DMA
    DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
    DMA_InitStructure.DMA_Memory0BaseAddr = (uint32_t)(&spi_slave_dma_buffer_out);
    DMA_InitStructure.DMA_BufferSize = BUF_OUT_LEN;
    DMA_InitStructure.DMA_Channel = SPI_SLAVE_TX_DMA_CHANNEL;
    DMA_InitStructure.DMA_DIR = DMA_DIR_MemoryToPeripheral;
    DMA_Init(SPI_SLAVE_TX_DMA_STREAM, &DMA_InitStructure);

    // Configure RX DMA
    DMA_InitStructure.DMA_Mode = DMA_Mode_Normal;
    DMA_InitStructure.DMA_Memory0BaseAddr = (uint32_t)(&spi_slave_dma_buffer_in_0);
    DMA_InitStructure.DMA_BufferSize = BUF_IN_LEN;
    DMA_InitStructure.DMA_Channel = SPI_SLAVE_RX_DMA_CHANNEL;
    DMA_InitStructure.DMA_DIR = DMA_DIR_PeripheralToMemory;
    // Using DMA Double Buffer mode for SPI RX, set the Buffer 0 as the first used buffer
    //DMA_DoubleBufferModeConfig(SPI_SLAVE_RX_DMA_STREAM, (uint32_t)(&spi_slave_dma_buffer_in_1), DMA_Memory_0);
    //DMA_DoubleBufferModeCmd(SPI_SLAVE_RX_DMA_STREAM, ENABLE);
    DMA_Init(SPI_SLAVE_RX_DMA_STREAM, &DMA_InitStructure);

    // Configure interrupts
    NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 7;
    NVIC_InitStructure.NVIC_IRQChannelSubPriority = 0;
    NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;

    NVIC_InitStructure.NVIC_IRQChannel = SPI_SLAVE_TX_DMA_IRQ;
    NVIC_Init(&NVIC_InitStructure);

    NVIC_InitStructure.NVIC_IRQChannel = SPI_SLAVE_RX_DMA_IRQ;
    NVIC_Init(&NVIC_InitStructure);
}

static void spiSlaveInit(void)
{
    SPI_InitTypeDef SPI_InitStructure;

    SPI_SLAVE_CLK_INIT(SPI_SLAVE_CLK, ENABLE);

    SPI_I2S_DeInit(SPI_SLAVE);

    SPI_InitStructure.SPI_Direction = SPI_Direction_2Lines_FullDuplex;
    SPI_InitStructure.SPI_Mode = SPI_Mode_Slave;
    SPI_InitStructure.SPI_DataSize = SPI_DataSize_8b;
    SPI_InitStructure.SPI_CPOL = SPI_CPOL_Low;
    SPI_InitStructure.SPI_CPHA = SPI_CPHA_1Edge;
    SPI_InitStructure.SPI_NSS = SPI_NSS_Soft;
    SPI_InitStructure.SPI_FirstBit = SPI_FirstBit_MSB;
    SPI_InitStructure.SPI_CRCPolynomial = 0; // Not used

    SPI_Init(SPI_SLAVE, &SPI_InitStructure);

}

static void spi_send_receive(uint8_t *tx_data_p, uint8_t *rx_data_p, uint16_t size)
{
    // Write to DMAy Streamx NDTR register 
    SPI_SLAVE_TX_DMA_STREAM->NDTR = size;
    // Write to DMAy Streamx M0AR 
    SPI_SLAVE_TX_DMA_STREAM->M0AR = (uint32_t) tx_data_p;
    // Write to DMAy Streamx NDTR register 
    SPI_SLAVE_RX_DMA_STREAM->NDTR = size;
    // Write to DMAy Streamx M0AR 
    SPI_SLAVE_RX_DMA_STREAM->M0AR = (uint32_t) rx_data_p;

    SPI_NSSInternalSoftwareConfig(SPI_SLAVE, SPI_NSSInternalSoft_Reset);
    DMA_Cmd(SPI_SLAVE_TX_DMA_STREAM, ENABLE);
    SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Tx, ENABLE);
    DMA_Cmd(SPI_SLAVE_RX_DMA_STREAM, ENABLE);
    SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Rx, ENABLE);
    // wait until txCompleteInterrupt has been called (gives the txComplete semaphore). This allows freeRTOS to do other tasks in the meantime
    while (pdTRUE != xSemaphoreTake(txComplete, portMAX_DELAY)) {;}
    // wait until rxCompleteInterrupt has been called (gives the rxComplete semaphore). This allows freeRTOS to do other tasks in the meantime
    while (pdTRUE != xSemaphoreTake(rxComplete, portMAX_DELAY)) {;}

    DMA_Cmd(SPI_SLAVE_RX_DMA_STREAM, DISABLE);
    SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Rx, DISABLE);
    DMA_Cmd(SPI_SLAVE_TX_DMA_STREAM, DISABLE);
    SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Tx, DISABLE);
    SPI_NSSInternalSoftwareConfig(SPI_SLAVE, SPI_NSSInternalSoft_Set);
}

static void spi_send(uint8_t *tx_data_p, uint16_t size)
{
  ASSERT_DMA_SAFE(tx_data_p);
  spi_send_receive(tx_data_p, spi_slave_dma_buffer_out, size);
  /*
  // Write to DMAy Streamx NDTR register 
  SPI_SLAVE_TX_DMA_STREAM->NDTR = size;
  // Write to DMAy Streamx M0AR 
  SPI_SLAVE_TX_DMA_STREAM->M0AR = (uint32_t) tx_data_p;

  SPI_NSSInternalSoftwareConfig(SPI_SLAVE, SPI_NSSInternalSoft_Reset);
  DMA_Cmd(SPI_SLAVE_TX_DMA_STREAM, ENABLE);
  SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Tx, ENABLE);
  // wait until txCompleteInterrupt has been called (gives the txComplete semaphore). This allows freeRTOS to do other tasks in the meantime
  while (pdTRUE != xSemaphoreTake(txComplete, portMAX_DELAY)) {;}
  DMA_Cmd(SPI_SLAVE_TX_DMA_STREAM, DISABLE);
  SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Tx, DISABLE);
  SPI_NSSInternalSoftwareConfig(SPI_SLAVE, SPI_NSSInternalSoft_Set);
  */
}

static void spi_receive(uint8_t *rx_data_p, uint16_t size)
{
  ASSERT_DMA_SAFE(rx_data_p);
  spi_send_receive(spi_slave_dma_buffer_out, rx_data_p, size);
  /*
  // Write to DMAy Streamx NDTR register 
  SPI_SLAVE_RX_DMA_STREAM->NDTR = size;
  // Write to DMAy Streamx M0AR 
  SPI_SLAVE_RX_DMA_STREAM->M0AR = (uint32_t) rx_data_p;
  // Software Pull-Down the NSS (CS) for always receiving
  SPI_NSSInternalSoftwareConfig(SPI_SLAVE, SPI_NSSInternalSoft_Reset);

  DMA_Cmd(SPI_SLAVE_RX_DMA_STREAM, ENABLE);
  SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Rx, ENABLE);

  // wait until rxCompleteInterrupt has been called (gives the rxComplete semaphore). This allows freeRTOS to do other tasks in the meantime
  while (pdTRUE != xSemaphoreTake(rxComplete, portMAX_DELAY)) {;}

  DMA_Cmd(SPI_SLAVE_RX_DMA_STREAM, DISABLE);
  SPI_I2S_DMACmd(SPI_SLAVE, SPI_I2S_DMAReq_Rx, DISABLE);

  SPI_NSSInternalSoftwareConfig(SPI_SLAVE, SPI_NSSInternalSoft_Set);
  */
}

static void spi_wait()
{
	// because this ist the slave do nothing.
}

static void spiSlaveTask(void *param)
{
    systemWaitStart();
    spiSlaveBegin();
    xSemaphoreGive(txComplete);
    xSemaphoreTake(txComplete, portMAX_DELAY);
    xSemaphoreGive(rxComplete);
    xSemaphoreTake(rxComplete, portMAX_DELAY);
    cf_state_machine_handle hstate_machine;
    init_cf_state_machine(&hstate_machine, round_finished, spi_wait, cp_connected_callback, get_cf_state, start_crazyflie, land_crazyflie, cf_launch_status, cf_land_status, init_position, get_current_trajectory, set_current_trajectory, &spi_send, &spi_receive, &spi_wait, &set_state_directly, &get_setpoint_state, &mocapSystemActive);
    run_cf_state_machine(&hstate_machine);
}

/*void get_init_pos(uint8_t id, float *init_pos)
{
    for (int i = 0; i < NUM_DRONES; i++) {
        if (drones_ids[i] == id) {
            init_pos[0] = drones_init_pos[i][0];
            init_pos[1] = drones_init_pos[i][1];
            init_pos[2] = drones_init_pos[i][2];
        }
    }
}*/

/******************** Privat Fun implement ********************/
void spiSlaveTaskInit()
{
    // TODO: PRI festzustellen in config.h
    STATIC_MEM_TASK_CREATE(spiSlaveTask, spiSlaveTask, SPI_SLAVE_TASK_NAME, NULL, SPI_SLAVE_TASK_PRI);
    isInit = true;
}

bool spiSlaveTaskTest()
{
    return isInit;
}

/******************** IRQ Fun implement ********************/
/**
 * @brief The IRQ Function are for the DMA transfer complete interrupt.
 *
 */
void __attribute__((used)) SPI_SLAVE_RX_DMA_IRQHandler(void)
{
    if (DMA_GetITStatus(SPI_SLAVE_RX_DMA_STREAM, DMA_IT_TCIF0) == SET)
    {
        portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

        DMA_ClearITPendingBit(SPI_SLAVE_RX_DMA_STREAM, SPI_SLAVE_RX_DMA_FLAG_TCIF);
        // Clear stream flags
        DMA_ClearFlag(SPI_SLAVE_RX_DMA_STREAM, SPI_SLAVE_RX_DMA_FLAG_TCIF);

        xSemaphoreGiveFromISR(rxComplete, &xHigherPriorityTaskWoken);
        if (xHigherPriorityTaskWoken)
        {
            portYIELD();
        }
    }
}

void __attribute__((used)) SPI_SLAVE_TX_DMA_IRQHandler(void)
{
    if (DMA_GetITStatus(SPI_SLAVE_TX_DMA_STREAM, DMA_IT_TCIF5) == SET)
    {
        portBASE_TYPE xHigherPriorityTaskWoken = pdFALSE;

        DMA_ClearITPendingBit(SPI_SLAVE_TX_DMA_STREAM, SPI_SLAVE_TX_DMA_FLAG_TCIF);
        // Clear stream flags
        DMA_ClearFlag(SPI_SLAVE_TX_DMA_STREAM, SPI_SLAVE_TX_DMA_FLAG_TCIF);

        xSemaphoreGiveFromISR(txComplete, &xHigherPriorityTaskWoken);
        if (xHigherPriorityTaskWoken)
        {
            portYIELD();
        }
    }
}

void spiSlaveMoCapSystemConnectedCallback(uint8_t heartBeat)
{
    mocapReady = true;
}

bool mocapSystemActive()
{
    return mocapReady;
}