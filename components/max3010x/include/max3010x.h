/**
 * @file max3010x.h
 * @brief MAX3010x Pulse Oximeter and Heart Rate Sensor Driver
 * 
 * This driver supports the MAX30102, MAX30105, and other devices in the MAX3010x family.
 * It provides hardware abstraction for I2C communication, sensor configuration,
 * and data acquisition from the pulse oximetry sensor.
 * 
 * HARDWARE FEATURES SUPPORTED:
 * - Dual LED (Red + IR) photoplethysmography
 * - 18-bit ADC with programmable sample rate
 * - 32-sample FIFO for burst data collection
 * - Integrated temperature sensor
 * - Proximity detection
 * - Configurable LED current and pulse width
 * - Multiple sampling rates (50Hz to 3200Hz)
 * 
 * SENSOR SPECIFICATIONS:
 * - Operating voltage: 1.8V (digital), 3.3V-5.0V (LED supply)
 * - I2C interface: Standard mode (100kHz) to Fast mode (400kHz)
 * - LED wavelengths: 660nm (Red), 880nm (IR)
 * - ADC resolution: 18-bit
 * - Sample rate: 50-3200 samples per second
 * 
 * @author Y3X Innovatech
 * @date 2025
 */

#ifndef MAX3010X_H
#define MAX3010X_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "driver/i2c.h"

/* ============================================================================
 * DEVICE IDENTIFICATION
 * ============================================================================ */

/** @brief MAX30102 I2C address */
#define MAX3010X_I2C_ADDRESS            0x57

/** @brief Expected Part IDs for different devices */
#define MAX30102_PART_ID                0x15
#define MAX30105_PART_ID                0x15

/** @brief Default I2C configuration */
#define MAX3010X_I2C_FREQ_HZ            400000
#define MAX3010X_I2C_TIMEOUT_MS         1000

/* ============================================================================
 * REGISTER ADDRESSES
 * ============================================================================ */

/** @brief Interrupt Status Registers */
#define MAX3010X_REG_INT_STATUS_1       0x00    ///< Main interrupt flags
#define MAX3010X_REG_INT_STATUS_2       0x01    ///< Secondary interrupt flags

/** @brief Interrupt Enable Registers */
#define MAX3010X_REG_INT_ENABLE_1       0x02    ///< Main interrupt enable
#define MAX3010X_REG_INT_ENABLE_2       0x03    ///< Secondary interrupt enable

/** @brief FIFO Registers */
#define MAX3010X_REG_FIFO_WRITE_PTR     0x04    ///< FIFO write pointer
#define MAX3010X_REG_FIFO_OVERFLOW      0x05    ///< FIFO overflow counter
#define MAX3010X_REG_FIFO_READ_PTR      0x06    ///< FIFO read pointer
#define MAX3010X_REG_FIFO_DATA          0x07    ///< FIFO data register

/** @brief Configuration Registers */
#define MAX3010X_REG_FIFO_CONFIG        0x08    ///< FIFO configuration
#define MAX3010X_REG_MODE_CONFIG        0x09    ///< Mode configuration
#define MAX3010X_REG_SPO2_CONFIG        0x0A    ///< SpO2 configuration
#define MAX3010X_REG_RESERVED_0B        0x0B    ///< Reserved

/** @brief LED Pulse Amplitude Registers */
#define MAX3010X_REG_LED1_PA            0x0C    ///< RED LED pulse amplitude
#define MAX3010X_REG_LED2_PA            0x0D    ///< IR LED pulse amplitude
#define MAX3010X_REG_LED3_PA            0x0E    ///< GREEN LED pulse amplitude (MAX30105 only)
#define MAX3010X_REG_RESERVED_0F        0x0F    ///< Reserved
#define MAX3010X_REG_LED_PROX_PA        0x10    ///< Proximity LED pulse amplitude

/** @brief Multi-LED Mode Registers */
#define MAX3010X_REG_MULTI_LED_1        0x11    ///< Multi-LED slot configuration 1
#define MAX3010X_REG_MULTI_LED_2        0x12    ///< Multi-LED slot configuration 2

/** @brief Temperature Registers */
#define MAX3010X_REG_TEMP_INT           0x1F    ///< Temperature integer part
#define MAX3010X_REG_TEMP_FRAC          0x20    ///< Temperature fractional part
#define MAX3010X_REG_TEMP_CONFIG        0x21    ///< Temperature configuration

/** @brief Proximity Detection */
#define MAX3010X_REG_PROX_INT_THRESH    0x30    ///< Proximity interrupt threshold

/** @brief Device Identification */
#define MAX3010X_REG_REV_ID             0xFE    ///< Revision ID
#define MAX3010X_REG_PART_ID            0xFF    ///< Part ID

/* ============================================================================
 * REGISTER BIT DEFINITIONS
 * ============================================================================ */

/** @brief Interrupt Status 1 (0x00) */
#define MAX3010X_INT_A_FULL             0x80    ///< FIFO almost full
#define MAX3010X_INT_DATA_RDY           0x40    ///< New FIFO data ready
#define MAX3010X_INT_ALC_OVF            0x20    ///< Ambient light cancellation overflow
#define MAX3010X_INT_PWR_RDY            0x01    ///< Power ready

/** @brief Interrupt Status 2 (0x01) */
#define MAX3010X_INT_DIE_TEMP_RDY       0x02    ///< Internal temperature ready

/** @brief FIFO Configuration (0x08) */
#define MAX3010X_FIFO_SMP_AVE_MASK      0xE0    ///< Sample averaging mask
#define MAX3010X_FIFO_SMP_AVE_1         0x00    ///< No averaging
#define MAX3010X_FIFO_SMP_AVE_2         0x20    ///< 2 samples averaged
#define MAX3010X_FIFO_SMP_AVE_4         0x40    ///< 4 samples averaged
#define MAX3010X_FIFO_SMP_AVE_8         0x60    ///< 8 samples averaged
#define MAX3010X_FIFO_SMP_AVE_16        0x80    ///< 16 samples averaged
#define MAX3010X_FIFO_SMP_AVE_32        0xA0    ///< 32 samples averaged

#define MAX3010X_FIFO_ROLLOVER_EN       0x10    ///< FIFO rollover enable
#define MAX3010X_FIFO_A_FULL_MASK       0x0F    ///< FIFO almost full threshold

/** @brief Mode Configuration (0x09) */
#define MAX3010X_MODE_SHDN              0x80    ///< Shutdown mode
#define MAX3010X_MODE_RESET             0x40    ///< Software reset
#define MAX3010X_MODE_MASK              0x07    ///< Mode mask

#define MAX3010X_MODE_HR_ONLY           0x02    ///< Heart rate only (RED LED)
#define MAX3010X_MODE_SPO2              0x03    ///< SpO2 mode (RED + IR LEDs)
#define MAX3010X_MODE_MULTI_LED         0x07    ///< Multi-LED mode

/** @brief SpO2 Configuration (0x0A) */
#define MAX3010X_SPO2_ADC_RGE_MASK      0x60    ///< ADC range mask
#define MAX3010X_SPO2_ADC_RGE_2048      0x00    ///< 2048 nA full scale
#define MAX3010X_SPO2_ADC_RGE_4096      0x20    ///< 4096 nA full scale
#define MAX3010X_SPO2_ADC_RGE_8192      0x40    ///< 8192 nA full scale
#define MAX3010X_SPO2_ADC_RGE_16384     0x60    ///< 16384 nA full scale

#define MAX3010X_SPO2_SR_MASK           0x1C    ///< Sample rate mask
#define MAX3010X_SPO2_SR_50             0x00    ///< 50 samples per second
#define MAX3010X_SPO2_SR_100            0x04    ///< 100 samples per second
#define MAX3010X_SPO2_SR_200            0x08    ///< 200 samples per second
#define MAX3010X_SPO2_SR_400            0x0C    ///< 400 samples per second
#define MAX3010X_SPO2_SR_800            0x10    ///< 800 samples per second
#define MAX3010X_SPO2_SR_1000           0x14    ///< 1000 samples per second
#define MAX3010X_SPO2_SR_1600           0x18    ///< 1600 samples per second
#define MAX3010X_SPO2_SR_3200           0x1C    ///< 3200 samples per second

#define MAX3010X_SPO2_PW_MASK           0x03    ///< Pulse width mask
#define MAX3010X_SPO2_PW_69US           0x00    ///< 69μs pulse width (15-bit ADC)
#define MAX3010X_SPO2_PW_118US          0x01    ///< 118μs pulse width (16-bit ADC)
#define MAX3010X_SPO2_PW_215US          0x02    ///< 215μs pulse width (17-bit ADC)
#define MAX3010X_SPO2_PW_411US          0x03    ///< 411μs pulse width (18-bit ADC)

/** @brief Multi-LED Slot Configuration */
#define MAX3010X_SLOT_DISABLED          0x00    ///< Slot disabled
#define MAX3010X_SLOT_RED_LED           0x01    ///< RED LED
#define MAX3010X_SLOT_IR_LED            0x02    ///< IR LED
#define MAX3010X_SLOT_GREEN_LED         0x03    ///< GREEN LED (MAX30105 only)
#define MAX3010X_SLOT_RESERVED_4        0x04    ///< Reserved
#define MAX3010X_SLOT_RESERVED_5        0x05    ///< Reserved
#define MAX3010X_SLOT_RESERVED_6        0x06    ///< Reserved
#define MAX3010X_SLOT_RED_PILOT         0x07    ///< RED LED pilot mode

/** @brief Temperature Configuration (0x21) */
#define MAX3010X_TEMP_EN                0x01    ///< Temperature enable

/* ============================================================================
 * DEVICE CONSTANTS
 * ============================================================================ */

/** @brief FIFO depth */
#define MAX3010X_FIFO_DEPTH             32

/** @brief Maximum number of LED channels */
#define MAX3010X_MAX_LED_CHANNELS       3

/** @brief Temperature sensor constants */
#define MAX3010X_TEMP_RESOLUTION        0.0625f ///< Temperature resolution in °C

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

/**
 * @brief LED current settings
 * 
 * LED current is programmable from 0mA to 50mA in approximately 0.2mA steps.
 * The actual current depends on the LED supply voltage and device characteristics.
 */
typedef enum {
    MAX3010X_LED_CURRENT_0MA    = 0x00,     ///< 0.0mA
    MAX3010X_LED_CURRENT_4_4MA  = 0x0F,     ///< ~4.4mA
    MAX3010X_LED_CURRENT_7_6MA  = 0x1F,     ///< ~7.6mA
    MAX3010X_LED_CURRENT_11MA   = 0x2F,     ///< ~11.0mA
    MAX3010X_LED_CURRENT_14_2MA = 0x3F,     ///< ~14.2mA
    MAX3010X_LED_CURRENT_17_4MA = 0x4F,     ///< ~17.4mA
    MAX3010X_LED_CURRENT_20_8MA = 0x5F,     ///< ~20.8mA
    MAX3010X_LED_CURRENT_24MA   = 0x6F,     ///< ~24.0mA
    MAX3010X_LED_CURRENT_27_1MA = 0x7F,     ///< ~27.1mA
    MAX3010X_LED_CURRENT_30_6MA = 0x8F,     ///< ~30.6mA
    MAX3010X_LED_CURRENT_33_8MA = 0x9F,     ///< ~33.8mA
    MAX3010X_LED_CURRENT_37MA   = 0xAF,     ///< ~37.0mA
    MAX3010X_LED_CURRENT_40_2MA = 0xBF,     ///< ~40.2mA
    MAX3010X_LED_CURRENT_43_6MA = 0xCF,     ///< ~43.6mA
    MAX3010X_LED_CURRENT_46_8MA = 0xDF,     ///< ~46.8mA
    MAX3010X_LED_CURRENT_50MA   = 0xFF      ///< ~50.0mA
} max3010x_led_current_t;

/**
 * @brief Sample rate settings
 * 
 * Higher sample rates provide better temporal resolution but consume more power
 * and generate more data. Choose based on application requirements.
 */
typedef enum {
    MAX3010X_SAMPLE_RATE_50HZ   = MAX3010X_SPO2_SR_50,    ///< 50 Hz sampling
    MAX3010X_SAMPLE_RATE_100HZ  = MAX3010X_SPO2_SR_100,   ///< 100 Hz sampling
    MAX3010X_SAMPLE_RATE_200HZ  = MAX3010X_SPO2_SR_200,   ///< 200 Hz sampling
    MAX3010X_SAMPLE_RATE_400HZ  = MAX3010X_SPO2_SR_400,   ///< 400 Hz sampling
    MAX3010X_SAMPLE_RATE_800HZ  = MAX3010X_SPO2_SR_800,   ///< 800 Hz sampling
    MAX3010X_SAMPLE_RATE_1000HZ = MAX3010X_SPO2_SR_1000,  ///< 1000 Hz sampling
    MAX3010X_SAMPLE_RATE_1600HZ = MAX3010X_SPO2_SR_1600,  ///< 1600 Hz sampling
    MAX3010X_SAMPLE_RATE_3200HZ = MAX3010X_SPO2_SR_3200   ///< 3200 Hz sampling
} max3010x_sample_rate_t;

/**
 * @brief Pulse width settings
 * 
 * Longer pulse width provides higher resolution and better SNR at the cost
 * of increased power consumption and reduced maximum sample rate.
 */
typedef enum {
    MAX3010X_PULSE_WIDTH_69US   = MAX3010X_SPO2_PW_69US,   ///< 69μs (15-bit)
    MAX3010X_PULSE_WIDTH_118US  = MAX3010X_SPO2_PW_118US,  ///< 118μs (16-bit)
    MAX3010X_PULSE_WIDTH_215US  = MAX3010X_SPO2_PW_215US,  ///< 215μs (17-bit)
    MAX3010X_PULSE_WIDTH_411US  = MAX3010X_SPO2_PW_411US   ///< 411μs (18-bit)
} max3010x_pulse_width_t;

/**
 * @brief ADC range settings
 * 
 * Larger ranges provide higher dynamic range but reduced sensitivity.
 * Choose based on expected signal levels and LED current settings.
 */
typedef enum {
    MAX3010X_ADC_RANGE_2048NA   = MAX3010X_SPO2_ADC_RGE_2048,   ///< 2048 nA
    MAX3010X_ADC_RANGE_4096NA   = MAX3010X_SPO2_ADC_RGE_4096,   ///< 4096 nA
    MAX3010X_ADC_RANGE_8192NA   = MAX3010X_SPO2_ADC_RGE_8192,   ///< 8192 nA
    MAX3010X_ADC_RANGE_16384NA  = MAX3010X_SPO2_ADC_RGE_16384   ///< 16384 nA
} max3010x_adc_range_t;

/**
 * @brief Operating modes
 */
typedef enum {
    MAX3010X_MODE_HEART_RATE    = MAX3010X_MODE_HR_ONLY,    ///< Heart rate only
    MAX3010X_MODE_SPO2_HR       = MAX3010X_MODE_SPO2,       ///< SpO2 + Heart rate
    MAX3010X_MODE_MULTI_LED_MODE = MAX3010X_MODE_MULTI_LED   ///< Multi-LED mode
} max3010x_mode_t;

/**
 * @brief Sample averaging settings
 */
typedef enum {
    MAX3010X_SAMPLE_AVG_1       = MAX3010X_FIFO_SMP_AVE_1,   ///< No averaging
    MAX3010X_SAMPLE_AVG_2       = MAX3010X_FIFO_SMP_AVE_2,   ///< 2 samples
    MAX3010X_SAMPLE_AVG_4       = MAX3010X_FIFO_SMP_AVE_4,   ///< 4 samples
    MAX3010X_SAMPLE_AVG_8       = MAX3010X_FIFO_SMP_AVE_8,   ///< 8 samples
    MAX3010X_SAMPLE_AVG_16      = MAX3010X_FIFO_SMP_AVE_16,  ///< 16 samples
    MAX3010X_SAMPLE_AVG_32      = MAX3010X_FIFO_SMP_AVE_32   ///< 32 samples
} max3010x_sample_avg_t;

/**
 * @brief Sensor configuration structure
 */
typedef struct {
    max3010x_mode_t mode;                   ///< Operating mode
    max3010x_sample_rate_t sample_rate;     ///< Sample rate
    max3010x_pulse_width_t pulse_width;     ///< LED pulse width
    max3010x_adc_range_t adc_range;         ///< ADC range
    max3010x_sample_avg_t sample_averaging; ///< Sample averaging
    max3010x_led_current_t red_led_current; ///< RED LED current
    max3010x_led_current_t ir_led_current;  ///< IR LED current
    max3010x_led_current_t green_led_current; ///< GREEN LED current (MAX30105)
    bool fifo_rollover_enable;              ///< Enable FIFO rollover
    uint8_t fifo_almost_full_threshold;     ///< FIFO almost full threshold (0-15)
} max3010x_config_t;

/**
 * @brief Device handle structure
 */
typedef struct {
    i2c_port_t i2c_port;                   ///< I2C port number
    uint8_t i2c_address;                    ///< I2C device address
    uint8_t part_id;                        ///< Device part ID
    uint8_t revision_id;                    ///< Device revision ID
    max3010x_config_t config;               ///< Current configuration
    bool initialized;                       ///< Initialization status
} max3010x_handle_t;

/**
 * @brief Raw sensor data structure
 */
typedef struct {
    uint32_t red;                           ///< RED LED photodiode reading
    uint32_t ir;                            ///< IR LED photodiode reading
    uint32_t green;                         ///< GREEN LED photodiode reading (MAX30105)
    uint32_t timestamp_ms;                  ///< Sample timestamp
} max3010x_raw_data_t;

/**
 * @brief FIFO status structure
 */
typedef struct {
    uint8_t write_pointer;                  ///< FIFO write pointer
    uint8_t read_pointer;                   ///< FIFO read pointer
    uint8_t overflow_counter;               ///< Number of lost samples
    uint8_t available_samples;              ///< Available samples to read
} max3010x_fifo_status_t;

/* ============================================================================
 * FUNCTION PROTOTYPES
 * ============================================================================ */

/**
 * @brief Initialize MAX3010x device
 * 
 * This function initializes the I2C communication and sets up the device
 * with default configuration. It verifies device presence and reads
 * identification registers.
 * 
 * @param[out] handle Device handle structure
 * @param[in] i2c_port I2C port number
 * @param[in] i2c_address Device I2C address (typically 0x57)
 * @return 
 *     - ESP_OK: Success
 *     - ESP_ERR_NOT_FOUND: Device not found on I2C bus
 *     - ESP_ERR_INVALID_ARG: Invalid parameters
 *     - Other ESP_ERR codes from I2C driver
 */
esp_err_t max3010x_init(max3010x_handle_t *handle, i2c_port_t i2c_port, uint8_t i2c_address);

/**
 * @brief Configure sensor with specified parameters
 * 
 * Applies a complete configuration to the sensor including mode, sample rate,
 * LED currents, and FIFO settings. This should be called after initialization
 * and before starting measurements.
 * 
 * @param[in] handle Device handle
 * @param[in] config Configuration structure
 * @return 
 *     - ESP_OK: Success
 *     - ESP_ERR_INVALID_ARG: Invalid configuration parameters
 *     - Other ESP_ERR codes from I2C operations
 */
esp_err_t max3010x_configure(max3010x_handle_t *handle, const max3010x_config_t *config);

/**
 * @brief Get default configuration
 * 
 * Returns a default configuration suitable for heart rate and SpO2 measurement.
 * This can be used as a starting point and modified as needed.
 * 
 * @param[out] config Configuration structure to fill
 */
void max3010x_get_default_config(max3010x_config_t *config);

/**
 * @brief Software reset the device
 * 
 * Performs a software reset which clears all configuration registers
 * and returns the device to power-on state.
 * 
 * @param[in] handle Device handle
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_reset(max3010x_handle_t *handle);

/**
 * @brief Enable or disable shutdown mode
 * 
 * In shutdown mode, the device stops all measurements and enters
 * low-power state while maintaining register contents.
 * 
 * @param[in] handle Device handle
 * @param[in] shutdown true to enter shutdown, false to wake up
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_set_shutdown(max3010x_handle_t *handle, bool shutdown);

/**
 * @brief Read single sample from FIFO
 * 
 * Reads one sample from the FIFO. The number of channels depends on
 * the configured mode (1 for HR mode, 2 for SpO2 mode, up to 4 for multi-LED).
 * 
 * @param[in] handle Device handle
 * @param[out] data Raw sensor data structure
 * @return 
 *     - ESP_OK: Success
 *     - ESP_ERR_NOT_FOUND: No data available in FIFO
 *     - Other ESP_ERR codes from I2C operations
 */
esp_err_t max3010x_read_sample(max3010x_handle_t *handle, max3010x_raw_data_t *data);

/**
 * @brief Read multiple samples from FIFO
 * 
 * Reads up to the specified number of samples from the FIFO.
 * The actual number read is returned in samples_read.
 * 
 * @param[in] handle Device handle
 * @param[out] data Array to store sensor data
 * @param[in] max_samples Maximum number of samples to read
 * @param[out] samples_read Actual number of samples read
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_read_samples(max3010x_handle_t *handle, max3010x_raw_data_t *data, 
                               uint8_t max_samples, uint8_t *samples_read);

/**
 * @brief Get FIFO status
 * 
 * Returns current FIFO pointers and available sample count.
 * Useful for determining when to read data and detecting overflows.
 * 
 * @param[in] handle Device handle
 * @param[out] status FIFO status structure
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_get_fifo_status(max3010x_handle_t *handle, max3010x_fifo_status_t *status);

/**
 * @brief Clear FIFO
 * 
 * Resets FIFO pointers to clear all stored samples.
 * Useful for starting fresh measurements.
 * 
 * @param[in] handle Device handle
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_clear_fifo(max3010x_handle_t *handle);

/**
 * @brief Read die temperature
 * 
 * Triggers a temperature measurement and returns the result.
 * Temperature reading takes approximately 29ms to complete.
 * 
 * @param[in] handle Device handle
 * @param[out] temperature Temperature in degrees Celsius
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_read_temperature(max3010x_handle_t *handle, float *temperature);

/**
 * @brief Configure interrupt settings
 * 
 * Enables or disables various interrupt sources. The interrupt pin
 * will assert when any enabled condition occurs.
 * 
 * @param[in] handle Device handle
 * @param[in] enable_fifo_almost_full Enable FIFO almost full interrupt
 * @param[in] enable_data_ready Enable new data ready interrupt
 * @param[in] enable_alc_overflow Enable ambient light cancellation overflow
 * @param[in] enable_temp_ready Enable temperature ready interrupt
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_configure_interrupts(max3010x_handle_t *handle,
                                       bool enable_fifo_almost_full,
                                       bool enable_data_ready,
                                       bool enable_alc_overflow,
                                       bool enable_temp_ready);

/**
 * @brief Read and clear interrupt status
 * 
 * Returns current interrupt flags and clears them by reading.
 * Should be called in interrupt service routine or polling loop.
 * 
 * @param[in] handle Device handle
 * @param[out] int_status_1 Main interrupt status register
 * @param[out] int_status_2 Secondary interrupt status register
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_get_interrupt_status(max3010x_handle_t *handle, 
                                       uint8_t *int_status_1, uint8_t *int_status_2);

/**
 * @brief Set LED current for specific LED
 * 
 * Configures the pulse amplitude (current) for individual LEDs.
 * Higher currents provide stronger signals but consume more power.
 * 
 * @param[in] handle Device handle
 * @param[in] led_number LED number (1=RED, 2=IR, 3=GREEN)
 * @param[in] current LED current setting
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_set_led_current(max3010x_handle_t *handle, uint8_t led_number, 
                                  max3010x_led_current_t current);

/**
 * @brief Configure proximity detection
 * 
 * Sets up proximity detection using ambient light measurement.
 * When an object is detected, proximity interrupt is generated.
 * 
 * @param[in] handle Device handle
 * @param[in] threshold Proximity threshold (0-255)
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_set_proximity_threshold(max3010x_handle_t *handle, uint8_t threshold);

/**
 * @brief Read device identification
 * 
 * Returns part ID and revision ID for device verification.
 * 
 * @param[in] handle Device handle
 * @param[out] part_id Device part ID
 * @param[out] revision_id Device revision ID
 * @return ESP_OK on success, or error code
 */
esp_err_t max3010x_get_device_id(max3010x_handle_t *handle, uint8_t *part_id, uint8_t *revision_id);

/**
 * @brief Check if device is present and responding
 * 
 * Performs a simple communication test to verify device presence.
 * 
 * @param[in] handle Device handle
 * @return true if device responds correctly, false otherwise
 */
bool max3010x_is_device_connected(max3010x_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif /* MAX3010X_H */