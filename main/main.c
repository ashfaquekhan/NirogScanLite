#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

static const char *TAG = "MAX30102";

// I2C Configuration for ESP32-S3-PICO
// Common I2C pins for ESP32-S3: GPIO8 (SDA) and GPIO9 (SCL)
// You may need to adjust these based on your wiring
#define I2C_MASTER_SCL_IO           9       // GPIO for I2C SCL
#define I2C_MASTER_SDA_IO           8       // GPIO for I2C SDA
#define I2C_MASTER_NUM              I2C_NUM_0
#define I2C_MASTER_FREQ_HZ          400000  // 400kHz
#define I2C_MASTER_TIMEOUT_MS       1000

// MAX30102 I2C Address (7-bit)
#define MAX30102_I2C_ADDR           0x57

// MAX30102 Register Addresses
#define MAX30102_INT_STATUS_1       0x00
#define MAX30102_INT_STATUS_2       0x01
#define MAX30102_INT_ENABLE_1       0x02
#define MAX30102_INT_ENABLE_2       0x03
#define MAX30102_FIFO_WRITE_PTR     0x04
#define MAX30102_FIFO_OVERFLOW      0x05
#define MAX30102_FIFO_READ_PTR      0x06
#define MAX30102_FIFO_DATA          0x07
#define MAX30102_FIFO_CONFIG        0x08
#define MAX30102_MODE_CONFIG        0x09
#define MAX30102_SPO2_CONFIG        0x0A
#define MAX30102_LED1_PA            0x0C  // RED LED
#define MAX30102_LED2_PA            0x0D  // IR LED
#define MAX30102_MULTI_LED_1        0x11
#define MAX30102_MULTI_LED_2        0x12
#define MAX30102_TEMP_INT           0x1F
#define MAX30102_TEMP_FRAC          0x20
#define MAX30102_TEMP_CONFIG        0x21
#define MAX30102_REV_ID             0xFE
#define MAX30102_PART_ID            0xFF

// Configuration Values
#define MAX30102_EXPECTED_PART_ID   0x15
#define MAX30102_RESET              0x40
#define MAX30102_MODE_SPO2          0x03  // SpO2 mode (RED + IR)
#define MAX30102_MODE_HR            0x02  // Heart rate mode (RED only)

// Sample averaging
#define MAX30102_SAMPLEAVG_1        0x00
#define MAX30102_SAMPLEAVG_2        0x20
#define MAX30102_SAMPLEAVG_4        0x40
#define MAX30102_SAMPLEAVG_8        0x60
#define MAX30102_SAMPLEAVG_16       0x80
#define MAX30102_SAMPLEAVG_32       0xA0

// FIFO rollover
#define MAX30102_ROLLOVER_ENABLE    0x10

// ADC Range
#define MAX30102_ADC_RANGE_2048     0x00
#define MAX30102_ADC_RANGE_4096     0x20
#define MAX30102_ADC_RANGE_8192     0x40
#define MAX30102_ADC_RANGE_16384    0x60

// Sample Rate
#define MAX30102_SAMPLERATE_50      0x00
#define MAX30102_SAMPLERATE_100     0x04
#define MAX30102_SAMPLERATE_200     0x08
#define MAX30102_SAMPLERATE_400     0x0C
#define MAX30102_SAMPLERATE_800     0x10
#define MAX30102_SAMPLERATE_1000    0x14
#define MAX30102_SAMPLERATE_1600    0x18
#define MAX30102_SAMPLERATE_3200    0x1C

// Pulse Width
#define MAX30102_PULSEWIDTH_69      0x00
#define MAX30102_PULSEWIDTH_118     0x01
#define MAX30102_PULSEWIDTH_215     0x02
#define MAX30102_PULSEWIDTH_411     0x03

// Multi-LED Mode Slots
#define SLOT_NONE                   0x00
#define SLOT_RED_LED                0x01
#define SLOT_IR_LED                 0x02

// Function prototypes
static esp_err_t i2c_master_init(void);
static esp_err_t max30102_write_register(uint8_t reg_addr, uint8_t data);
static esp_err_t max30102_read_register(uint8_t reg_addr, uint8_t *data);
static esp_err_t max30102_read_registers(uint8_t reg_addr, uint8_t *data, size_t len);
static esp_err_t max30102_modify_register(uint8_t reg_addr, uint8_t mask, uint8_t value);
static esp_err_t max30102_verify_device(void);
static esp_err_t max30102_soft_reset(void);
static esp_err_t max30102_clear_fifo(void);
static esp_err_t max30102_setup(uint8_t power_level, uint8_t sample_avg, 
                                 uint8_t led_mode, uint8_t sample_rate, 
                                 uint8_t pulse_width, uint8_t adc_range);
static esp_err_t max30102_read_fifo(uint32_t *red_led, uint32_t *ir_led);
static esp_err_t max30102_read_temperature(float *temperature);

/**
 * @brief Initialize I2C master
 */
static esp_err_t i2c_master_init(void)
{
    ESP_LOGI(TAG, "Initializing I2C on SDA=%d, SCL=%d", I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO);
    
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
        .clk_flags = 0,  // ESP32-S3 specific
    };
    
    esp_err_t err = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C param config failed: %s", esp_err_to_name(err));
        return err;
    }
    
    err = i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(err));
        return err;
    }
    
    ESP_LOGI(TAG, "I2C initialized successfully");
    return ESP_OK;
}

/**
 * @brief Write a byte to MAX30102 register
 */
static esp_err_t max30102_write_register(uint8_t reg_addr, uint8_t data)
{
    uint8_t write_buf[2] = {reg_addr, data};
    
    esp_err_t ret = i2c_master_write_to_device(I2C_MASTER_NUM, MAX30102_I2C_ADDR,
                                                write_buf, sizeof(write_buf),
                                                pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to write register 0x%02X: %s", reg_addr, esp_err_to_name(ret));
    }
    
    return ret;
}

/**
 * @brief Read a byte from MAX30102 register
 */
static esp_err_t max30102_read_register(uint8_t reg_addr, uint8_t *data)
{
    esp_err_t ret = i2c_master_write_read_device(I2C_MASTER_NUM, MAX30102_I2C_ADDR,
                                                   &reg_addr, 1, data, 1,
                                                   pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read register 0x%02X: %s", reg_addr, esp_err_to_name(ret));
    }
    
    return ret;
}

/**
 * @brief Read multiple bytes from MAX30102 registers
 */
static esp_err_t max30102_read_registers(uint8_t reg_addr, uint8_t *data, size_t len)
{
    esp_err_t ret = i2c_master_write_read_device(I2C_MASTER_NUM, MAX30102_I2C_ADDR,
                                                   &reg_addr, 1, data, len,
                                                   pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to read %d bytes from register 0x%02X: %s", 
                 len, reg_addr, esp_err_to_name(ret));
    }
    
    return ret;
}

/**
 * @brief Modify specific bits in a register (read-modify-write)
 */
static esp_err_t max30102_modify_register(uint8_t reg_addr, uint8_t mask, uint8_t value)
{
    uint8_t reg_value;
    esp_err_t ret;
    
    ret = max30102_read_register(reg_addr, &reg_value);
    if (ret != ESP_OK) {
        return ret;
    }
    
    reg_value = (reg_value & mask) | value;
    
    return max30102_write_register(reg_addr, reg_value);
}

/**
 * @brief Verify MAX30102 device by reading Part ID
 */
static esp_err_t max30102_verify_device(void)
{
    uint8_t part_id, rev_id;
    esp_err_t ret;
    
    ret = max30102_read_register(MAX30102_PART_ID, &part_id);
    if (ret != ESP_OK) {
        return ret;
    }
    
    ret = max30102_read_register(MAX30102_REV_ID, &rev_id);
    if (ret != ESP_OK) {
        return ret;
    }
    
    ESP_LOGI(TAG, "Part ID: 0x%02X, Revision ID: 0x%02X", part_id, rev_id);
    
    if (part_id != MAX30102_EXPECTED_PART_ID) {
        ESP_LOGE(TAG, "Unexpected Part ID! Expected 0x%02X, got 0x%02X",
                 MAX30102_EXPECTED_PART_ID, part_id);
        return ESP_ERR_INVALID_VERSION;
    }
    
    ESP_LOGI(TAG, "MAX30102 verified successfully");
    return ESP_OK;
}

/**
 * @brief Perform soft reset of MAX30102
 */
static esp_err_t max30102_soft_reset(void)
{
    esp_err_t ret;
    
    ESP_LOGI(TAG, "Performing soft reset...");
    ret = max30102_write_register(MAX30102_MODE_CONFIG, MAX30102_RESET);
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Wait for reset to complete
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Verify reset completed by checking if reset bit cleared
    uint8_t mode_config;
    ret = max30102_read_register(MAX30102_MODE_CONFIG, &mode_config);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Soft reset complete. Mode config: 0x%02X", mode_config);
    }
    
    return ret;
}

/**
 * @brief Clear FIFO pointers
 */
static esp_err_t max30102_clear_fifo(void)
{
    esp_err_t ret;
    
    ret = max30102_write_register(MAX30102_FIFO_WRITE_PTR, 0x00);
    if (ret != ESP_OK) return ret;
    
    ret = max30102_write_register(MAX30102_FIFO_OVERFLOW, 0x00);
    if (ret != ESP_OK) return ret;
    
    ret = max30102_write_register(MAX30102_FIFO_READ_PTR, 0x00);
    if (ret != ESP_OK) return ret;
    
    ESP_LOGI(TAG, "FIFO cleared");
    return ESP_OK;
}

/**
 * @brief Setup MAX30102 with specified configuration
 */
static esp_err_t max30102_setup(uint8_t power_level, uint8_t sample_avg, 
                                 uint8_t led_mode, uint8_t sample_rate, 
                                 uint8_t pulse_width, uint8_t adc_range)
{
    esp_err_t ret;
    
    ESP_LOGI(TAG, "Configuring MAX30102...");
    
    // Perform soft reset
    ret = max30102_soft_reset();
    if (ret != ESP_OK) return ret;
    
    // Configure FIFO
    // Sample averaging + FIFO rollover enabled + FIFO almost full = 0
    uint8_t fifo_config = sample_avg | MAX30102_ROLLOVER_ENABLE | 0x00;
    ret = max30102_write_register(MAX30102_FIFO_CONFIG, fifo_config);
    if (ret != ESP_OK) return ret;
    ESP_LOGI(TAG, "FIFO config: 0x%02X", fifo_config);
    
    // Set mode (SpO2 or HR)
    ret = max30102_write_register(MAX30102_MODE_CONFIG, led_mode);
    if (ret != ESP_OK) return ret;
    ESP_LOGI(TAG, "Mode config: 0x%02X", led_mode);
    
    // Configure SpO2 - ADC range, sample rate, pulse width
    uint8_t spo2_config = adc_range | sample_rate | pulse_width;
    ret = max30102_write_register(MAX30102_SPO2_CONFIG, spo2_config);
    if (ret != ESP_OK) return ret;
    ESP_LOGI(TAG, "SpO2 config: 0x%02X", spo2_config);
    
    // Set LED pulse amplitudes
    ret = max30102_write_register(MAX30102_LED1_PA, power_level);  // RED LED
    if (ret != ESP_OK) return ret;
    
    ret = max30102_write_register(MAX30102_LED2_PA, power_level);  // IR LED
    if (ret != ESP_OK) return ret;
    ESP_LOGI(TAG, "LED power level: 0x%02X", power_level);
    
    // Configure multi-LED mode slots (for SpO2 mode)
    if (led_mode == MAX30102_MODE_SPO2) {
        // Slot 1 = RED LED, Slot 2 = IR LED
        uint8_t multi_led_1 = (SLOT_IR_LED << 4) | SLOT_RED_LED;
        ret = max30102_write_register(MAX30102_MULTI_LED_1, multi_led_1);
        if (ret != ESP_OK) return ret;
        ESP_LOGI(TAG, "Multi-LED config: 0x%02X", multi_led_1);
    }
    
    // Clear FIFO
    ret = max30102_clear_fifo();
    if (ret != ESP_OK) return ret;
    
    ESP_LOGI(TAG, "MAX30102 configuration complete");
    return ESP_OK;
}

/**
 * @brief Read one sample from FIFO (RED and IR values)
 */
static esp_err_t max30102_read_fifo(uint32_t *red_led, uint32_t *ir_led)
{
    uint8_t fifo_data[6];  // 3 bytes per LED channel (RED + IR = 6 bytes)
    esp_err_t ret;
    
    // Read 6 bytes from FIFO
    ret = max30102_read_registers(MAX30102_FIFO_DATA, fifo_data, 6);
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Combine 3 bytes into 18-bit value for RED LED
    // Data is left-justified, so we keep the upper 18 bits
    *red_led = ((uint32_t)fifo_data[0] << 16) | 
               ((uint32_t)fifo_data[1] << 8) | 
               ((uint32_t)fifo_data[2]);
    *red_led &= 0x3FFFF;  // Mask to 18 bits
    
    // Combine 3 bytes into 18-bit value for IR LED
    *ir_led = ((uint32_t)fifo_data[3] << 16) | 
              ((uint32_t)fifo_data[4] << 8) | 
              ((uint32_t)fifo_data[5]);
    *ir_led &= 0x3FFFF;  // Mask to 18 bits
    
    return ESP_OK;
}

/**
 * @brief Read temperature from MAX30102
 */
static esp_err_t max30102_read_temperature(float *temperature)
{
    uint8_t temp_int, temp_frac, temp_config;
    esp_err_t ret;
    int timeout = 100;  // 100ms timeout
    
    // Trigger temperature measurement
    ret = max30102_write_register(MAX30102_TEMP_CONFIG, 0x01);
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Wait for temperature measurement to complete
    // Poll TEMP_CONFIG register until bit 0 clears
    do {
        vTaskDelay(pdMS_TO_TICKS(10));
        ret = max30102_read_register(MAX30102_TEMP_CONFIG, &temp_config);
        if (ret != ESP_OK) {
            return ret;
        }
        timeout -= 10;
    } while ((temp_config & 0x01) && (timeout > 0));
    
    if (timeout <= 0) {
        ESP_LOGE(TAG, "Temperature measurement timeout");
        return ESP_ERR_TIMEOUT;
    }
    
    // Read temperature integer part
    ret = max30102_read_register(MAX30102_TEMP_INT, &temp_int);
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Read temperature fractional part
    ret = max30102_read_register(MAX30102_TEMP_FRAC, &temp_frac);
    if (ret != ESP_OK) {
        return ret;
    }
    
    // Calculate temperature in Celsius
    // Integer part is in 2's complement, fractional part in increments of 0.0625°C
    int8_t temp_int_signed = (int8_t)temp_int;
    float temp_frac_float = (temp_frac & 0x0F) * 0.0625f;
    *temperature = (float)temp_int_signed + temp_frac_float;
    
    return ESP_OK;
}

/**
 * @brief Main application task
 */
void app_main(void)
{
    esp_err_t ret;
    
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "MAX30102 Sensor Example for ESP32-S3-PICO");
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "I2C SDA: GPIO%d, SCL: GPIO%d", I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO);
    ESP_LOGI(TAG, "==============================================");
    
    // Initialize I2C
    ret = i2c_master_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "I2C initialization failed!");
        ESP_LOGE(TAG, "Please check:");
        ESP_LOGE(TAG, "  1. Wiring: SDA=GPIO%d, SCL=GPIO%d", I2C_MASTER_SDA_IO, I2C_MASTER_SCL_IO);
        ESP_LOGE(TAG, "  2. Pull-up resistors (typically 4.7k ohm)");
        ESP_LOGE(TAG, "  3. Sensor power supply (1.8V and 3.3V)");
        return;
    }
    
    // Give sensor time to power up
    vTaskDelay(pdMS_TO_TICKS(100));
    
    // Verify device
    ret = max30102_verify_device();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Device verification failed!");
        ESP_LOGE(TAG, "Please check:");
        ESP_LOGE(TAG, "  1. I2C connections");
        ESP_LOGE(TAG, "  2. Sensor address (should be 0x57)");
        ESP_LOGE(TAG, "  3. Power supply to sensor");
        return;
    }
    
    // Setup sensor
    // Parameters: power_level (0x1F=6.4mA), sample_avg (4 samples), 
    //             led_mode (SpO2), sample_rate (100 samples/sec),
    //             pulse_width (411µs/18-bit), adc_range (4096nA)
    ret = max30102_setup(
        0x1F,                       // Power level (LED current)
        MAX30102_SAMPLEAVG_4,       // Average 4 samples
        MAX30102_MODE_SPO2,         // SpO2 mode (RED + IR)
        MAX30102_SAMPLERATE_100,    // 100 samples per second
        MAX30102_PULSEWIDTH_411,    // 411µs pulse width (18-bit resolution)
        MAX30102_ADC_RANGE_4096     // 4096nA full scale
    );
    
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Sensor setup failed!");
        return;
    }
    
    ESP_LOGI(TAG, "\n=== Starting measurements ===\n");
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    // Main loop - read sensor data
    uint32_t sample_count = 0;
    while (1) {
        uint32_t red_value = 0, ir_value = 0;
        float temperature = 0.0f;
        
        // Read FIFO data (RED and IR)
        ret = max30102_read_fifo(&red_value, &ir_value);
        if (ret == ESP_OK) {
            sample_count++;
            
            // Read temperature every 10 samples (to avoid excessive polling)
            if (sample_count % 10 == 0) {
                ret = max30102_read_temperature(&temperature);
                if (ret == ESP_OK) {
                    ESP_LOGI(TAG, "Sample %5lu | RED: %6lu | IR: %6lu | Temp: %5.2f°C",
                             sample_count, red_value, ir_value, temperature);
                } else {
                    ESP_LOGE(TAG, "Temperature read failed!");
                }
            } else {
                ESP_LOGI(TAG, "Sample %5lu | RED: %6lu | IR: %6lu",
                         sample_count, red_value, ir_value);
            }
        } else {
            ESP_LOGE(TAG, "FIFO read failed!");
        }
        
        // Delay between readings (100 samples/sec = 10ms per sample)
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}