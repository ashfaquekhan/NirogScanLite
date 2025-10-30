#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/i2c.h"

#include "max3010x.h"
#include "max30102_algorithms.h"

static const char *TAG = "MAX30102";

/* ============================================================================
 * CONFIGURATION
 * ============================================================================ */

// I2C Configuration
#define I2C_SCL_IO              9
#define I2C_SDA_IO              8
#define I2C_PORT                I2C_NUM_0
#define I2C_FREQ_HZ             400000

// Sampling Configuration  
#define SAMPLE_RATE_HZ          100
#define SAMPLE_INTERVAL_MS      10
#define SPO2_BUFFER_SIZE        400     // 4 seconds
#define TEMP_READ_INTERVAL      200     // Every 2 seconds

// Signal thresholds
#define MIN_SIGNAL              10000   // Too weak
#define MAX_SIGNAL              200000  // Saturation threshold
#define GOOD_SIGNAL_MIN         30000   // Good signal range
#define GOOD_SIGNAL_MAX         150000

/* ============================================================================
 * GLOBALS
 * ============================================================================ */

static max3010x_handle_t sensor;
static max30102_hr_state_t hr_state;

static uint32_t red_buffer[SPO2_BUFFER_SIZE];
static uint32_t ir_buffer[SPO2_BUFFER_SIZE];
static uint16_t buffer_idx = 0;

static float hr = 0.0f;
static float spo2 = 0.0f;
static float temp = 25.0f;

static uint8_t current_led_current = MAX3010X_LED_CURRENT_11MA;  // Start conservative

/* ============================================================================
 * FUNCTIONS
 * ============================================================================ */

static void init_i2c(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_SDA_IO,
        .scl_io_num = I2C_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
        .clk_flags = 0,
    };
    i2c_param_config(I2C_PORT, &conf);
    i2c_driver_install(I2C_PORT, conf.mode, 0, 0, 0);
    ESP_LOGI(TAG, "I2C initialized");
}

static void init_sensor(void) {
    ESP_LOGI(TAG, "Initializing MAX30102...");
    
    // Initialize sensor
    if (max3010x_init(&sensor, I2C_PORT, MAX3010X_I2C_ADDRESS) != ESP_OK) {
        ESP_LOGE(TAG, "Sensor init failed!");
        return;
    }
    
    // Configure sensor
    max3010x_config_t config;
    max3010x_get_default_config(&config);
    
    config.mode = MAX3010X_MODE_SPO2_HR;
    config.sample_rate = MAX3010X_SAMPLE_RATE_100HZ;
    config.pulse_width = MAX3010X_PULSE_WIDTH_411US;        // 18-bit
    config.adc_range = MAX3010X_ADC_RANGE_4096NA;
    config.sample_averaging = MAX3010X_SAMPLE_AVG_1;
    config.red_led_current = current_led_current;           // Start conservative
    config.ir_led_current = current_led_current;
    config.green_led_current = MAX3010X_LED_CURRENT_0MA;
    config.fifo_rollover_enable = true;
    config.fifo_almost_full_threshold = 15;                 // Leave room
    
    if (max3010x_configure(&sensor, &config) != ESP_OK) {
        ESP_LOGE(TAG, "Sensor config failed!");
        return;
    }
    
    // Clear FIFO to start fresh
    max3010x_clear_fifo(&sensor);
    
    // Initialize algorithm
    max30102_hr_init(&hr_state);
    
    ESP_LOGI(TAG, "MAX30102 initialized successfully");
}

static void adjust_led_current(uint32_t ir_signal) {
    static uint32_t adjust_counter = 0;
    
    // Only adjust every 50 samples to avoid instability
    if (++adjust_counter < 50) return;
    adjust_counter = 0;
    
    uint8_t new_current = current_led_current;
    
    if (ir_signal < MIN_SIGNAL) {
        // Signal too weak - increase current
        if (current_led_current < MAX3010X_LED_CURRENT_27_1MA) {
            new_current = current_led_current + 0x10;  // Increase by one step
            ESP_LOGI(TAG, "Signal weak (%lu), increasing LED current", ir_signal);
        }
    } else if (ir_signal > MAX_SIGNAL) {
        // Signal saturating - decrease current
        if (current_led_current > MAX3010X_LED_CURRENT_4_4MA) {
            new_current = current_led_current - 0x10;  // Decrease by one step
            ESP_LOGI(TAG, "Signal saturated (%lu), decreasing LED current", ir_signal);
        }
    }
    
    if (new_current != current_led_current) {
        current_led_current = new_current;
        // Apply new current
        max3010x_set_led_current(&sensor, 1, current_led_current);  // RED LED
        max3010x_set_led_current(&sensor, 2, current_led_current);  // IR LED
    }
}

static void process_spo2(void) {
    if (buffer_idx < SPO2_BUFFER_SIZE) return;
    
    max30102_spo2_result_t result;
    max30102_calculate_spo2(ir_buffer, SPO2_BUFFER_SIZE, red_buffer, &result);
    
    // Update values if valid
    if (result.hr_valid && result.heart_rate >= 40 && result.heart_rate <= 200) {
        hr = (float)result.heart_rate;
    }
    if (result.spo2_valid && result.spo2_value >= 80 && result.spo2_value <= 100) {
        spo2 = (float)result.spo2_value;
    }
    
    // Slide buffer - keep last 75%
    uint16_t keep = SPO2_BUFFER_SIZE * 3 / 4;
    memmove(red_buffer, &red_buffer[SPO2_BUFFER_SIZE - keep], keep * sizeof(uint32_t));
    memmove(ir_buffer, &ir_buffer[SPO2_BUFFER_SIZE - keep], keep * sizeof(uint32_t));
    buffer_idx = keep;
}

static bool read_sample_safe(max3010x_raw_data_t *sample) {
    // Check FIFO status first
    max3010x_fifo_status_t fifo_status;
    if (max3010x_get_fifo_status(&sensor, &fifo_status) != ESP_OK) {
        return false;
    }
    
    // Clear FIFO if it's getting too full (prevents sticking)
    if (fifo_status.available_samples > 20) {
        ESP_LOGW(TAG, "FIFO overflow (%d samples), clearing", fifo_status.available_samples);
        max3010x_clear_fifo(&sensor);
        return false;
    }
    
    // Read sample if available
    if (fifo_status.available_samples > 0) {
        return (max3010x_read_sample(&sensor, sample) == ESP_OK);
    }
    
    return false;
}

/* ============================================================================
 * MAIN APPLICATION
 * ============================================================================ */

void app_main(void) {
    ESP_LOGI(TAG, "MAX30102 Working Monitor: 100Hz | 18-bit | Auto LED Current");
    
    // Initialize
    init_i2c();
    init_sensor();
    
    // Give sensor time to stabilize
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    printf("Red_Raw,IR_Raw,PPG_Signal,HR,SpO2,Temp\n");
    ESP_LOGI(TAG, "Starting measurements...");
    
    TickType_t last_wake = xTaskGetTickCount();
    uint32_t sample_count = 0;
    uint32_t no_data_count = 0;
    
    while (1) {
        max3010x_raw_data_t sample;
        
        if (read_sample_safe(&sample)) {
            sample_count++;
            no_data_count = 0;  // Reset no-data counter
            
            // Auto-adjust LED current based on signal strength
            adjust_led_current(sample.ir);
            
            // Heart rate detection
            max30102_check_for_beat(&hr_state, sample.ir);
            int16_t ppg = hr_state.ir_ac_signal_current;
            
            // Store for SpO2 calculation
            if (buffer_idx < SPO2_BUFFER_SIZE) {
                red_buffer[buffer_idx] = sample.red;
                ir_buffer[buffer_idx] = sample.ir;
                buffer_idx++;
            }
            
            // Process SpO2
            process_spo2();
            
            // Read temperature periodically
            if (sample_count % TEMP_READ_INTERVAL == 0) {
                float temperature;
                if (max3010x_read_temperature(&sensor, &temperature) == ESP_OK) {
                    temp = temperature;
                }
            }
            
            // Output data
            printf("%lu,%lu,%d,%.1f,%.1f,%.2f\n",
                   sample.red, sample.ir, ppg, hr, spo2, temp);
                   
        } else {
            // No data available
            no_data_count++;
            
            // If no data for too long, clear FIFO
            if (no_data_count > 100) {
                ESP_LOGW(TAG, "No data for too long, clearing FIFO");
                max3010x_clear_fifo(&sensor);
                no_data_count = 0;
            }
        }
        
        // Status every 5 seconds
        if (sample_count > 0 && sample_count % 500 == 0) {
            ESP_LOGI(TAG, "Samples: %lu, HR: %.1f, SpO2: %.1f%%, Temp: %.2f°C", 
                     sample_count, hr, spo2, temp);
        }
        
        // Maintain timing
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(SAMPLE_INTERVAL_MS));
    }
}