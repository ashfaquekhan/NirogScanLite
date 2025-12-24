/**
 * @file nirog_scan_fixed.c
 * @brief Advanced Health Monitoring System for ESP32-S3 (Compilation Fixed)
 * 
 * Industrial-grade implementation featuring:
 * - ECG sampling at 125Hz via AD8232
 * - PPG sampling at 50Hz via MAX30101/102 
 * - Synchronized data collection (4 PPG + 5 ECG samples per 80ms packet)
 * - BLE GATT server for real-time data transmission
 * - Fuel gauge monitoring (MAX17048)
 * - Temperature sensing via MAX30101 internal sensor
 * - Leads-off detection for ECG quality monitoring
 * 
 * @author Embedded Systems Engineer
 * @version 1.0.1 (Fixed compilation issues)
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include "esp_system.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"

#include "nvs_flash.h"
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_bt_main.h"
#include "esp_gatt_common_api.h"

/* Compatibility macros for ESP_RETURN_ON_ERROR */
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifndef ESP_RETURN_ON_ERROR
#define ESP_RETURN_ON_ERROR(x, tag, format, ...) do { \
    esp_err_t err_rc_ = (x); \
    if (unlikely(err_rc_ != ESP_OK)) { \
        ESP_LOGE(tag, format, ##__VA_ARGS__); \
        return err_rc_; \
    } \
} while(0)
#endif

/*============================================================================
 * SYSTEM CONFIGURATION
 *============================================================================*/

#define TAG                           "NirogScan"
#define SYSTEM_VERSION                "1.0.1"

/* Sampling Configuration */
#define ECG_SAMPLE_RATE_HZ            125U
#define PPG_SAMPLE_RATE_HZ            50U
#define PACKET_RATE_HZ                12U  /* 80ms packets */

#define ECG_TIMER_PERIOD_US           (1000000U / ECG_SAMPLE_RATE_HZ)  /* 8000us */
#define PPG_TIMER_PERIOD_US           (1000000U / PPG_SAMPLE_RATE_HZ)  /* 20000us */
#define PACKET_TIMER_PERIOD_US        (1000000U / PACKET_RATE_HZ)      /* 83333us */

/* Packet Structure: 80ms window */
#define ECG_SAMPLES_PER_PACKET        10U  /* 125Hz * 0.08s = 10 */
#define PPG_SAMPLES_PER_PACKET        4U   /* 50Hz * 0.08s = 4 */

/* Hardware Pins */
#define I2C_MASTER_SCL_IO_PPG         9
#define I2C_MASTER_SDA_IO_PPG         8
#define I2C_MASTER_NUM_PPG            I2C_NUM_0

#define I2C_MASTER_SCL_IO_FG          11
#define I2C_MASTER_SDA_IO_FG          10  
#define I2C_MASTER_NUM_FG             I2C_NUM_1

#define I2C_FREQ_HZ                   400000U
#define I2C_TIMEOUT_MS                1000U

#define ECG_ADC_CHANNEL               ADC_CHANNEL_1
#define ECG_ADC_ATTEN                 ADC_ATTEN_DB_12
#define ECG_ADC_BITWIDTH              ADC_BITWIDTH_12
#define ECG_GPIO_NUM                  GPIO_NUM_2

#define LO_PLUS_PIN                   GPIO_NUM_16
#define LO_MINUS_PIN                  GPIO_NUM_17

/* Device Addresses */
#define MAX30101_I2C_ADDR             0x57
#define MAX17048_I2C_ADDR             0x36

/* MAX30101 Registers */
#define MAX30101_REG_INT_STATUS_1     0x00
#define MAX30101_REG_INT_STATUS_2     0x01
#define MAX30101_REG_INT_ENABLE_1     0x02
#define MAX30101_REG_INT_ENABLE_2     0x03
#define MAX30101_REG_FIFO_WR_PTR      0x04
#define MAX30101_REG_FIFO_OVF_COUNTER 0x05
#define MAX30101_REG_FIFO_RD_PTR      0x06
#define MAX30101_REG_FIFO_DATA        0x07
#define MAX30101_REG_FIFO_CONFIG      0x08
#define MAX30101_REG_MODE_CONFIG      0x09
#define MAX30101_REG_SPO2_CONFIG      0x0A
#define MAX30101_REG_LED1_PA          0x0C
#define MAX30101_REG_LED2_PA          0x0D
#define MAX30101_REG_DIETEMPINT       0x1F
#define MAX30101_REG_DIETEMPFRAC      0x20
#define MAX30101_REG_DIETEMPCONFIG    0x21
#define MAX30101_REG_PART_ID          0xFF

/* MAX17048 Registers */
#define MAX17048_REG_VCELL            0x02
#define MAX17048_REG_SOC              0x04

/* BLE Configuration */
#define DEVICE_NAME                   "NirogScan"
#define HEALTH_SERVICE_UUID           0x180D  /* Heart Rate Service */
#define DATA_CHARACTERISTIC_UUID      0x2A37  /* Heart Rate Measurement */
#define BATTERY_SERVICE_UUID          0x180F  /* Battery Service */
#define BATTERY_LEVEL_UUID            0x2A19  /* Battery Level */

/* Queue and Buffer Sizes */
#define ECG_QUEUE_SIZE                32U
#define PPG_QUEUE_SIZE                16U
#define PACKET_QUEUE_SIZE             8U
#define BLE_DATA_QUEUE_SIZE           4U

/* System Limits */
#define MAX_BLE_PACKET_SIZE           512U
#define TEMP_READ_INTERVAL_MS         5000U
#define BATTERY_READ_INTERVAL_MS      10000U

/*============================================================================
 * TYPE DEFINITIONS  
 *============================================================================*/

typedef enum {
    SYSTEM_STATE_INIT = 0,
    SYSTEM_STATE_RUNNING,
    SYSTEM_STATE_ERROR,
    SYSTEM_STATE_SHUTDOWN
} system_state_t;

typedef struct {
    int16_t value;
    uint8_t leads_off_status;  /* bit0: LO+, bit1: LO- */
    uint32_t timestamp_us;
} ecg_sample_t;

typedef struct {
    uint32_t red;
    uint32_t ir; 
    uint32_t timestamp_us;
} ppg_sample_t;

typedef struct {
    float voltage;
    float percentage;
    uint32_t timestamp_us;
} battery_data_t;

typedef struct {
    float temperature;
    uint32_t timestamp_us;
} temperature_data_t;

/**
 * @brief Unified health data packet (80ms window)
 * 
 * Contains synchronized samples from all sensors:
 * - 10 ECG samples (125Hz * 0.08s)
 * - 4 PPG samples (50Hz * 0.08s)
 * - System data (battery, temperature, leads-off)
 */
typedef struct {
    /* Header */
    uint16_t sequence_number;
    uint32_t timestamp_start_us;
    uint32_t timestamp_end_us;
    
    /* ECG Data (10 samples) */
    int16_t ecg_values[ECG_SAMPLES_PER_PACKET];
    uint8_t leads_off_status[ECG_SAMPLES_PER_PACKET];
    
    /* PPG Data (4 samples) */
    uint32_t ppg_red[PPG_SAMPLES_PER_PACKET];
    uint32_t ppg_ir[PPG_SAMPLES_PER_PACKET];
    
    /* System Data */
    float battery_voltage;
    float battery_percentage;
    float temperature;
    
    /* Status Flags */
    uint8_t ecg_quality;     /* 0-100% */
    uint8_t ppg_quality;     /* 0-100% */
    uint8_t system_status;   /* bit flags for errors */
    
    /* Footer for integrity */
    uint16_t checksum;
} health_data_packet_t;

typedef struct {
    esp_gatts_cb_t gatts_cb;
    uint16_t gatts_if;
    uint16_t app_id;
    uint16_t conn_id;
    uint16_t service_handle;
    uint16_t char_handle;
    uint16_t descr_handle;
    esp_bt_uuid_t service_uuid;
    esp_bt_uuid_t char_uuid;
} ble_profile_t;

/*============================================================================
 * GLOBAL VARIABLES
 *============================================================================*/

/* Hardware Handles */
static adc_oneshot_unit_handle_t adc1_handle = NULL;
static esp_timer_handle_t ecg_timer = NULL;
static esp_timer_handle_t ppg_timer = NULL;
static esp_timer_handle_t packet_timer = NULL;

/* FreeRTOS Objects */
static QueueHandle_t ecg_queue = NULL;
static QueueHandle_t ppg_queue = NULL;
static QueueHandle_t packet_queue = NULL;
static QueueHandle_t ble_data_queue = NULL;
static SemaphoreHandle_t sensor_data_mutex = NULL;

/* System State */
static volatile system_state_t system_state = SYSTEM_STATE_INIT;
static volatile uint16_t packet_sequence = 0;

/* Sensor Data Buffers */
static ecg_sample_t ecg_buffer[ECG_SAMPLES_PER_PACKET];
static ppg_sample_t ppg_buffer[PPG_SAMPLES_PER_PACKET];
static battery_data_t battery_data = {0};
static temperature_data_t temp_data = {0};

/* BLE Profile */
static ble_profile_t health_profile = {0};
static bool ble_connected = false;
static uint16_t ble_conn_id = 0;

/* Statistics */
static uint32_t ecg_sample_count = 0;
static uint32_t ppg_sample_count = 0;
static uint32_t packet_count = 0;
static uint32_t ble_tx_count = 0;

/*============================================================================
 * UTILITY FUNCTIONS
 *============================================================================*/

/**
 * @brief Calculate simple checksum for data integrity
 */
static uint16_t calculate_checksum(const uint8_t *data, size_t length) {
    uint16_t checksum = 0;
    for (size_t i = 0; i < length; i++) {
        checksum += data[i];
    }
    return checksum;
}

/**
 * @brief Validate ECG signal quality based on leads-off detection
 */
static uint8_t calculate_ecg_quality(const uint8_t *leads_off_status, size_t count) {
    uint32_t good_samples = 0;
    for (size_t i = 0; i < count; i++) {
        if (leads_off_status[i] == 0) {
            good_samples++;
        }
    }
    return (uint8_t)((good_samples * 100) / count);
}

/**
 * @brief Validate PPG signal quality based on signal amplitude
 */
static uint8_t calculate_ppg_quality(const uint32_t *ir_values, size_t count) {
    uint32_t valid_samples = 0;
    for (size_t i = 0; i < count; i++) {
        if (ir_values[i] > 1000 && ir_values[i] < 262000) {
            valid_samples++;
        }
    }
    return (uint8_t)((valid_samples * 100) / count);
}

/*============================================================================
 * HARDWARE ABSTRACTION LAYER
 *============================================================================*/

/**
 * @brief Initialize I2C master interface
 */
static esp_err_t i2c_master_init(i2c_port_t port, int sda_pin, int scl_pin) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = sda_pin,
        .scl_io_num = scl_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_FREQ_HZ,
    };
    
    ESP_RETURN_ON_ERROR(i2c_param_config(port, &conf), TAG, "I2C param config failed");
    return i2c_driver_install(port, conf.mode, 0, 0, 0);
}

/**
 * @brief Write single register to I2C device
 */
static esp_err_t i2c_write_reg(i2c_port_t port, uint8_t device_addr, uint8_t reg, uint8_t data) {
    uint8_t write_buf[2] = {reg, data};
    return i2c_master_write_to_device(port, device_addr, write_buf, sizeof(write_buf), pdMS_TO_TICKS(I2C_TIMEOUT_MS));
}

/**
 * @brief Read single register from I2C device
 */
static esp_err_t i2c_read_reg(i2c_port_t port, uint8_t device_addr, uint8_t reg, uint8_t *data) {
    return i2c_master_write_read_device(port, device_addr, &reg, 1, data, 1, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
}

/**
 * @brief Read multiple registers from I2C device
 */
static esp_err_t i2c_read_regs(i2c_port_t port, uint8_t device_addr, uint8_t reg, uint8_t *data, size_t length) {
    return i2c_master_write_read_device(port, device_addr, &reg, 1, data, length, pdMS_TO_TICKS(I2C_TIMEOUT_MS));
}

/*============================================================================
 * MAX30101 DRIVER FUNCTIONS
 *============================================================================*/

/**
 * @brief Initialize MAX30101 pulse oximeter sensor
 */
static esp_err_t max30101_init(void) {
    uint8_t part_id;
    ESP_RETURN_ON_ERROR(i2c_read_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_PART_ID, &part_id), TAG, "Failed to read part ID");
    
    if (part_id != 0x15) {
        ESP_LOGE(TAG, "Invalid MAX30101 part ID: 0x%02X", part_id);
        return ESP_ERR_NOT_FOUND;
    }
    
    ESP_LOGI(TAG, "MAX30101 detected, Part ID: 0x%02X", part_id);
    
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_MODE_CONFIG, 0x40), TAG, "Reset failed");
    vTaskDelay(pdMS_TO_TICKS(100));
    
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_WR_PTR, 0x00), TAG, "FIFO WR PTR failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_OVF_COUNTER, 0x00), TAG, "FIFO OVF failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_RD_PTR, 0x00), TAG, "FIFO RD PTR failed");
    
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_CONFIG, 0x4F), TAG, "FIFO config failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_MODE_CONFIG, 0x03), TAG, "Mode config failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_SPO2_CONFIG, 0x03), TAG, "SpO2 config failed");
    
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_LED1_PA, 0x24), TAG, "LED1 config failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_LED2_PA, 0x24), TAG, "LED2 config failed");
    
    return ESP_OK;
}

/**
 * @brief Read FIFO data from MAX30101
 */
static esp_err_t max30101_read_fifo(uint32_t *red, uint32_t *ir) {
    uint8_t wr_ptr, rd_ptr;
    
    ESP_RETURN_ON_ERROR(i2c_read_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_WR_PTR, &wr_ptr), TAG, "Read WR PTR failed");
    ESP_RETURN_ON_ERROR(i2c_read_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_RD_PTR, &rd_ptr), TAG, "Read RD PTR failed");
    
    uint8_t samples_available = (wr_ptr - rd_ptr) & 0x1F;
    if (samples_available == 0) {
        return ESP_ERR_NOT_FOUND;
    }
    
    uint8_t fifo_data[6];
    ESP_RETURN_ON_ERROR(i2c_read_regs(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_FIFO_DATA, fifo_data, 6), TAG, "Read FIFO failed");
    
    *red = ((uint32_t)fifo_data[0] << 16) | ((uint32_t)fifo_data[1] << 8) | fifo_data[2];
    *red &= 0x3FFFF;
    
    *ir = ((uint32_t)fifo_data[3] << 16) | ((uint32_t)fifo_data[4] << 8) | fifo_data[5];
    *ir &= 0x3FFFF;
    
    return ESP_OK;
}

/**
 * @brief Read temperature from MAX30101 internal sensor
 */
static esp_err_t max30101_read_temperature(float *temperature) {
    ESP_RETURN_ON_ERROR(i2c_write_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_DIETEMPCONFIG, 0x01), TAG, "Temp config failed");
    
    /* Wait for temperature conversion */
    uint8_t status;
    for (int i = 0; i < 100; i++) {
        vTaskDelay(pdMS_TO_TICKS(1));
        ESP_RETURN_ON_ERROR(i2c_read_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_INT_STATUS_2, &status), TAG, "Status read failed");
        if (status & 0x02) break;
    }
    
    uint8_t temp_int, temp_frac;
    ESP_RETURN_ON_ERROR(i2c_read_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_DIETEMPINT, &temp_int), TAG, "Temp int read failed");
    ESP_RETURN_ON_ERROR(i2c_read_reg(I2C_MASTER_NUM_PPG, MAX30101_I2C_ADDR, MAX30101_REG_DIETEMPFRAC, &temp_frac), TAG, "Temp frac read failed");
    
    *temperature = (int8_t)temp_int + (temp_frac * 0.0625f);
    return ESP_OK;
}

/*============================================================================
 * MAX17048 FUEL GAUGE FUNCTIONS
 *============================================================================*/

/**
 * @brief Read battery data from MAX17048 fuel gauge
 */
static esp_err_t max17048_read_battery(float *voltage, float *percentage) {
    uint8_t buffer[2];
    
    ESP_RETURN_ON_ERROR(i2c_read_regs(I2C_MASTER_NUM_FG, MAX17048_I2C_ADDR, MAX17048_REG_VCELL, buffer, 2), TAG, "VCELL read failed");
    uint16_t vcell = (buffer[0] << 8) | buffer[1];
    *voltage = (vcell >> 4) * 1.25f / 1000.0f;
    
    ESP_RETURN_ON_ERROR(i2c_read_regs(I2C_MASTER_NUM_FG, MAX17048_I2C_ADDR, MAX17048_REG_SOC, buffer, 2), TAG, "SOC read failed");
    uint16_t soc = (buffer[0] << 8) | buffer[1];
    *percentage = (soc >> 8) + (soc & 0xFF) / 256.0f;
    
    return ESP_OK;
}

/*============================================================================
 * ADC FUNCTIONS FOR ECG
 *============================================================================*/

/**
 * @brief Initialize ADC for ECG signal acquisition
 */
static esp_err_t adc_init(void) {
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_RETURN_ON_ERROR(adc_oneshot_new_unit(&init_config, &adc1_handle), TAG, "ADC init failed");
    
    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ECG_ADC_BITWIDTH,
        .atten = ECG_ADC_ATTEN,
    };
    return adc_oneshot_config_channel(adc1_handle, ECG_ADC_CHANNEL, &config);
}

/*============================================================================
 * GPIO FUNCTIONS
 *============================================================================*/

/**
 * @brief Initialize GPIO for leads-off detection
 */
static esp_err_t gpio_init_leads_off(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LO_PLUS_PIN) | (1ULL << LO_MINUS_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    return gpio_config(&io_conf);
}

/*============================================================================
 * BLE IMPLEMENTATION
 *============================================================================*/

/**
 * @brief BLE GAP event handler
 */
static void gap_event_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    switch (event) {
        case ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT:
            esp_ble_gap_start_advertising(&(esp_ble_adv_params_t){
                .adv_int_min = 0x20,
                .adv_int_max = 0x40,
                .adv_type = ADV_TYPE_IND,
                .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
                .channel_map = ADV_CHNL_ALL,
                .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
            });
            break;
        default:
            break;
    }
}

/**
 * @brief BLE GATT server event handler  
 */
static void gatts_event_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, esp_ble_gatts_cb_param_t *param) {
    switch (event) {
        case ESP_GATTS_REG_EVT:
            if (param->reg.status == ESP_GATT_OK) {
                health_profile.gatts_if = gatts_if;
                
                esp_ble_gap_set_device_name(DEVICE_NAME);
                esp_ble_gap_config_adv_data(&(esp_ble_adv_data_t){
                    .set_scan_rsp = false,
                    .include_name = true,
                    .include_txpower = true,
                    .min_interval = 0x0006,
                    .max_interval = 0x0010,
                    .appearance = 0x00,
                    .manufacturer_len = 0,
                    .p_manufacturer_data = NULL,
                    .service_data_len = 0,
                    .p_service_data = NULL,
                    .service_uuid_len = 0,
                    .p_service_uuid = NULL,
                    .flag = (ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT),
                });
                
                esp_ble_gatts_create_service(gatts_if, &(esp_gatt_srvc_id_t){
                    .is_primary = true,
                    .id.inst_id = 0x00,
                    .id.uuid.len = ESP_UUID_LEN_16,
                    .id.uuid.uuid.uuid16 = HEALTH_SERVICE_UUID,
                }, 4);
            }
            break;
            
        case ESP_GATTS_CREATE_EVT:
            if (param->create.status == ESP_GATT_OK) {
                health_profile.service_handle = param->create.service_handle;
                esp_ble_gatts_start_service(health_profile.service_handle);
                
                esp_ble_gatts_add_char(health_profile.service_handle, &(esp_bt_uuid_t){
                    .len = ESP_UUID_LEN_16,
                    .uuid.uuid16 = DATA_CHARACTERISTIC_UUID,
                }, ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
                ESP_GATT_CHAR_PROP_BIT_READ | ESP_GATT_CHAR_PROP_BIT_NOTIFY,
                NULL, NULL);
            }
            break;
            
        case ESP_GATTS_ADD_CHAR_EVT:
            if (param->add_char.status == ESP_GATT_OK) {
                health_profile.char_handle = param->add_char.attr_handle;
            }
            break;
            
        case ESP_GATTS_CONNECT_EVT:
            ble_connected = true;
            ble_conn_id = param->connect.conn_id;
            ESP_LOGI(TAG, "BLE client connected");
            break;
            
        case ESP_GATTS_DISCONNECT_EVT:
            ble_connected = false;
            esp_ble_gap_start_advertising(&(esp_ble_adv_params_t){
                .adv_int_min = 0x20,
                .adv_int_max = 0x40,
                .adv_type = ADV_TYPE_IND,
                .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
                .channel_map = ADV_CHNL_ALL,
                .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
            });
            ESP_LOGI(TAG, "BLE client disconnected");
            break;
            
        default:
            break;
    }
}

/**
 * @brief Initialize BLE stack and GATT server
 */
static esp_err_t ble_init(void) {
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    ESP_RETURN_ON_ERROR(esp_bt_controller_init(&bt_cfg), TAG, "BT controller init failed");
    ESP_RETURN_ON_ERROR(esp_bt_controller_enable(ESP_BT_MODE_BLE), TAG, "BT controller enable failed");
    ESP_RETURN_ON_ERROR(esp_bluedroid_init(), TAG, "Bluedroid init failed");
    ESP_RETURN_ON_ERROR(esp_bluedroid_enable(), TAG, "Bluedroid enable failed");
    
    ESP_RETURN_ON_ERROR(esp_ble_gatts_register_callback(gatts_event_handler), TAG, "GATTS callback register failed");
    ESP_RETURN_ON_ERROR(esp_ble_gap_register_callback(gap_event_handler), TAG, "GAP callback register failed");
    ESP_RETURN_ON_ERROR(esp_ble_gatts_app_register(0), TAG, "GATTS app register failed");
    ESP_RETURN_ON_ERROR(esp_ble_gatt_set_local_mtu(512), TAG, "Set local MTU failed");
    
    return ESP_OK;
}

/*============================================================================
 * INTERRUPT SERVICE ROUTINES
 *============================================================================*/

/**
 * @brief ECG timer callback - 125Hz sampling
 */
static void IRAM_ATTR ecg_timer_callback(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    ecg_sample_t sample;
    
    int adc_raw;
    if (adc_oneshot_read(adc1_handle, ECG_ADC_CHANNEL, &adc_raw) == ESP_OK) {
        sample.value = (int16_t)adc_raw;
        sample.timestamp_us = esp_timer_get_time();
        
        sample.leads_off_status = 0;
        if (gpio_get_level(LO_PLUS_PIN)) sample.leads_off_status |= 0x01;
        if (gpio_get_level(LO_MINUS_PIN)) sample.leads_off_status |= 0x02;
        
        xQueueSendFromISR(ecg_queue, &sample, &xHigherPriorityTaskWoken);
        ecg_sample_count++;
    }
    
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

/**
 * @brief PPG timer callback - 50Hz sampling
 */
static void IRAM_ATTR ppg_timer_callback(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    ppg_sample_t sample;
    
    uint32_t red, ir;
    if (max30101_read_fifo(&red, &ir) == ESP_OK) {
        if (red > 1000 && ir > 1000) {
            sample.red = red;
            sample.ir = ir;
            sample.timestamp_us = esp_timer_get_time();
            
            xQueueSendFromISR(ppg_queue, &sample, &xHigherPriorityTaskWoken);
            ppg_sample_count++;
        }
    }
    
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

/**
 * @brief Packet timer callback - 12Hz packet generation
 */
static void IRAM_ATTR packet_timer_callback(void *arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    uint32_t trigger = 1;
    
    xQueueSendFromISR(packet_queue, &trigger, &xHigherPriorityTaskWoken);
    
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

/*============================================================================
 * TASK IMPLEMENTATIONS
 *============================================================================*/

/**
 * @brief System monitoring task
 */
static void system_monitor_task(void *pvParameters) {
    TickType_t last_temp_read = 0;
    TickType_t last_battery_read = 0;
    
    while (system_state == SYSTEM_STATE_RUNNING) {
        TickType_t current_time = xTaskGetTickCount();
        
        if ((current_time - last_temp_read) >= pdMS_TO_TICKS(TEMP_READ_INTERVAL_MS)) {
            float temperature;
            if (max30101_read_temperature(&temperature) == ESP_OK) {
                if (xSemaphoreTake(sensor_data_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                    temp_data.temperature = temperature;
                    temp_data.timestamp_us = esp_timer_get_time();
                    xSemaphoreGive(sensor_data_mutex);
                }
            }
            last_temp_read = current_time;
        }
        
        if ((current_time - last_battery_read) >= pdMS_TO_TICKS(BATTERY_READ_INTERVAL_MS)) {
            float voltage, percentage;
            if (max17048_read_battery(&voltage, &percentage) == ESP_OK) {
                if (xSemaphoreTake(sensor_data_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                    battery_data.voltage = voltage;
                    battery_data.percentage = percentage;
                    battery_data.timestamp_us = esp_timer_get_time();
                    xSemaphoreGive(sensor_data_mutex);
                }
            }
            last_battery_read = current_time;
        }
        
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
    
    vTaskDelete(NULL);
}

/**
 * @brief Data packet assembly and transmission task
 */
static void packet_assembly_task(void *pvParameters) {
    health_data_packet_t packet;
    uint32_t trigger;
    
    while (system_state == SYSTEM_STATE_RUNNING) {
        if (xQueueReceive(packet_queue, &trigger, portMAX_DELAY) == pdTRUE) {
            memset(&packet, 0, sizeof(packet));
            
            packet.sequence_number = packet_sequence++;
            packet.timestamp_start_us = esp_timer_get_time();
            
            /* Collect ECG samples */
            for (int i = 0; i < ECG_SAMPLES_PER_PACKET; i++) {
                if (xQueueReceive(ecg_queue, &ecg_buffer[i], pdMS_TO_TICKS(10)) == pdTRUE) {
                    packet.ecg_values[i] = ecg_buffer[i].value;
                    packet.leads_off_status[i] = ecg_buffer[i].leads_off_status;
                } else {
                    packet.ecg_values[i] = (i > 0) ? packet.ecg_values[i-1] : 0;
                    packet.leads_off_status[i] = 0xFF;
                }
            }
            
            /* Collect PPG samples */
            for (int i = 0; i < PPG_SAMPLES_PER_PACKET; i++) {
                if (xQueueReceive(ppg_queue, &ppg_buffer[i], pdMS_TO_TICKS(10)) == pdTRUE) {
                    packet.ppg_red[i] = ppg_buffer[i].red;
                    packet.ppg_ir[i] = ppg_buffer[i].ir;
                } else {
                    packet.ppg_red[i] = (i > 0) ? packet.ppg_red[i-1] : 0;
                    packet.ppg_ir[i] = (i > 0) ? packet.ppg_ir[i-1] : 0;
                }
            }
            
            /* Get system data */
            if (xSemaphoreTake(sensor_data_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                packet.battery_voltage = battery_data.voltage;
                packet.battery_percentage = battery_data.percentage;
                packet.temperature = temp_data.temperature;
                xSemaphoreGive(sensor_data_mutex);
            }
            
            packet.timestamp_end_us = esp_timer_get_time();
            
            /* Calculate quality metrics */
            packet.ecg_quality = calculate_ecg_quality(packet.leads_off_status, ECG_SAMPLES_PER_PACKET);
            
            /* Copy PPG IR data to avoid packed member address issue */
            uint32_t ppg_ir_temp[PPG_SAMPLES_PER_PACKET];
            memcpy(ppg_ir_temp, packet.ppg_ir, sizeof(ppg_ir_temp));
            packet.ppg_quality = calculate_ppg_quality(ppg_ir_temp, PPG_SAMPLES_PER_PACKET);
            
            packet.system_status = (system_state == SYSTEM_STATE_RUNNING) ? 0x00 : 0x01;
            
            /* Calculate checksum */
            packet.checksum = calculate_checksum((uint8_t*)&packet, sizeof(packet) - sizeof(packet.checksum));
            
            /* Send to BLE queue */
            if (ble_connected) {
                xQueueSend(ble_data_queue, &packet, 0);
            }
            
            /* Output to UART for debugging */
            printf("PKT>seq:%u,ecg_qual:%u,ppg_qual:%u,batt:%.2f,temp:%.2f,ts:%lu\n",
                   packet.sequence_number, packet.ecg_quality, packet.ppg_quality,
                   packet.battery_voltage, packet.temperature, 
                   (unsigned long)packet.timestamp_end_us);
            
            packet_count++;
        }
    }
    
    vTaskDelete(NULL);
}

/**
 * @brief BLE data transmission task
 */
static void ble_transmission_task(void *pvParameters) {
    health_data_packet_t packet;
    
    while (system_state == SYSTEM_STATE_RUNNING) {
        if (xQueueReceive(ble_data_queue, &packet, portMAX_DELAY) == pdTRUE) {
            if (ble_connected && health_profile.char_handle != 0) {
                esp_err_t ret = esp_ble_gatts_send_indicate(
                    health_profile.gatts_if, ble_conn_id, health_profile.char_handle,
                    sizeof(packet), (uint8_t*)&packet, false);
                
                if (ret == ESP_OK) {
                    ble_tx_count++;
                } else {
                    ESP_LOGW(TAG, "BLE transmission failed: %s", esp_err_to_name(ret));
                }
            }
        }
    }
    
    vTaskDelete(NULL);
}

/*============================================================================
 * SYSTEM INITIALIZATION
 *============================================================================*/

/**
 * @brief Initialize all hardware peripherals
 */
static esp_err_t hardware_init(void) {
    ESP_RETURN_ON_ERROR(gpio_init_leads_off(), TAG, "GPIO init failed");
    ESP_RETURN_ON_ERROR(adc_init(), TAG, "ADC init failed");
    ESP_RETURN_ON_ERROR(i2c_master_init(I2C_MASTER_NUM_PPG, I2C_MASTER_SDA_IO_PPG, I2C_MASTER_SCL_IO_PPG), TAG, "I2C PPG init failed");
    ESP_RETURN_ON_ERROR(i2c_master_init(I2C_MASTER_NUM_FG, I2C_MASTER_SDA_IO_FG, I2C_MASTER_SCL_IO_FG), TAG, "I2C FG init failed");
    ESP_RETURN_ON_ERROR(max30101_init(), TAG, "MAX30101 init failed");
    
    return ESP_OK;
}

/**
 * @brief Initialize FreeRTOS objects
 */
static esp_err_t rtos_init(void) {
    ecg_queue = xQueueCreate(ECG_QUEUE_SIZE, sizeof(ecg_sample_t));
    ppg_queue = xQueueCreate(PPG_QUEUE_SIZE, sizeof(ppg_sample_t));
    packet_queue = xQueueCreate(PACKET_QUEUE_SIZE, sizeof(uint32_t));
    ble_data_queue = xQueueCreate(BLE_DATA_QUEUE_SIZE, sizeof(health_data_packet_t));
    sensor_data_mutex = xSemaphoreCreateMutex();
    
    if (!ecg_queue || !ppg_queue || !packet_queue || !ble_data_queue || !sensor_data_mutex) {
        ESP_LOGE(TAG, "Failed to create FreeRTOS objects");
        return ESP_ERR_NO_MEM;
    }
    
    return ESP_OK;
}

/**
 * @brief Start all timer interrupts
 */
static esp_err_t timers_start(void) {
    const esp_timer_create_args_t ecg_timer_args = {
        .callback = &ecg_timer_callback,
        .name = "ecg_timer"
    };
    
    const esp_timer_create_args_t ppg_timer_args = {
        .callback = &ppg_timer_callback,
        .name = "ppg_timer"
    };
    
    const esp_timer_create_args_t packet_timer_args = {
        .callback = &packet_timer_callback,
        .name = "packet_timer"
    };
    
    ESP_RETURN_ON_ERROR(esp_timer_create(&ecg_timer_args, &ecg_timer), TAG, "ECG timer create failed");
    ESP_RETURN_ON_ERROR(esp_timer_create(&ppg_timer_args, &ppg_timer), TAG, "PPG timer create failed");
    ESP_RETURN_ON_ERROR(esp_timer_create(&packet_timer_args, &packet_timer), TAG, "Packet timer create failed");
    
    ESP_RETURN_ON_ERROR(esp_timer_start_periodic(ecg_timer, ECG_TIMER_PERIOD_US), TAG, "ECG timer start failed");
    ESP_RETURN_ON_ERROR(esp_timer_start_periodic(ppg_timer, PPG_TIMER_PERIOD_US), TAG, "PPG timer start failed");
    ESP_RETURN_ON_ERROR(esp_timer_start_periodic(packet_timer, PACKET_TIMER_PERIOD_US), TAG, "Packet timer start failed");
    
    return ESP_OK;
}

/*============================================================================
 * MAIN APPLICATION
 *============================================================================*/

void app_main(void) {
    ESP_LOGI(TAG, "NirogScan Health Monitor v%s Starting", SYSTEM_VERSION);
    ESP_LOGI(TAG, "ECG: %dHz | PPG: %dHz | Packets: %dHz", ECG_SAMPLE_RATE_HZ, PPG_SAMPLE_RATE_HZ, PACKET_RATE_HZ);
    
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);
    
    if (hardware_init() != ESP_OK) {
        ESP_LOGE(TAG, "Hardware initialization failed");
        return;
    }
    
    if (rtos_init() != ESP_OK) {
        ESP_LOGE(TAG, "FreeRTOS initialization failed");
        return;
    }
    
    if (ble_init() != ESP_OK) {
        ESP_LOGE(TAG, "BLE initialization failed");
        return;
    }
    
    system_state = SYSTEM_STATE_RUNNING;
    
    xTaskCreatePinnedToCore(system_monitor_task, "sys_monitor", 4096, NULL, 3, NULL, 0);
    xTaskCreatePinnedToCore(packet_assembly_task, "pkt_assembly", 8192, NULL, 2, NULL, 0);
    xTaskCreatePinnedToCore(ble_transmission_task, "ble_tx", 4096, NULL, 4, NULL, 1);
    
    if (timers_start() != ESP_OK) {
        ESP_LOGE(TAG, "Timer initialization failed");
        return;
    }
    
    ESP_LOGI(TAG, "System initialized successfully");
    
    /* Main monitoring loop */
    TickType_t last_stats = xTaskGetTickCount();
    
    while (system_state == SYSTEM_STATE_RUNNING) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        
        TickType_t current_time = xTaskGetTickCount();
        if ((current_time - last_stats) >= pdMS_TO_TICKS(30000)) {
            ESP_LOGI(TAG, "Stats: ECG=%lu, PPG=%lu, Packets=%lu, BLE_TX=%lu",
                     ecg_sample_count, ppg_sample_count, packet_count, ble_tx_count);
            
            ESP_LOGI(TAG, "Heap: Free=%lu, Min=%lu", 
                     (unsigned long)esp_get_free_heap_size(), 
                     (unsigned long)esp_get_minimum_free_heap_size());
            
            last_stats = current_time;
        }
    }
}