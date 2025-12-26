// ============================================================================
// NIROG HEALTH MONITOR - OPTIMIZED SINGLE-TASK ARCHITECTURE
// ECG 125Hz, PPG 50Hz, Packets 25Hz, BLE 25Hz
// ============================================================================

#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "nvs_flash.h"
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_bt_main.h"

#define TAG "Nirog"

// Sample rates
#define ECG_RATE_HZ          125
#define PPG_RATE_HZ          50
#define PACKET_RATE_HZ       25

#define MASTER_RATE_HZ       125
#define MASTER_PERIOD_US     (1000000 / MASTER_RATE_HZ)

// Samples per packet (perfect division: 125/25=5, 50/25=2)
#define ECG_PER_PKT          5
#define PPG_PER_PKT          2

// Pins
#define ECG_ADC_CH           ADC_CHANNEL_1
#define LO_PLUS              GPIO_NUM_16
#define LO_MINUS             GPIO_NUM_17
#define I2C_SCL_PPG          9
#define I2C_SDA_PPG          8
#define I2C_PORT_PPG         I2C_NUM_0
#define I2C_SCL_FG           11
#define I2C_SDA_FG           10
#define I2C_PORT_FG          I2C_NUM_1

// I2C addresses
#define MAX30101_ADDR        0x57
#define MAX17048_ADDR        0x36

#define DEVICE_NAME          "NirogScan"
#define PACKET_QUEUE_SIZE    4

// ============================================================================
// PACKET STRUCTURE
// ============================================================================

typedef struct __attribute__((packed)) {
    uint32_t timestamp;
    uint16_t seq;
    int16_t ecg[5];
    uint8_t leads;
    uint32_t red[2];
    uint32_t ir[2];
    float battery_v;
    float battery_pct;
    float temp;
    uint16_t crc;
} packet_t;

// ============================================================================
// GLOBALS
// ============================================================================

static adc_oneshot_unit_handle_t adc1 = NULL;
static esp_timer_handle_t master_timer = NULL;
static TaskHandle_t acq_task_h = NULL;
static TaskHandle_t ble_task_h = NULL;
static QueueHandle_t packet_queue = NULL;

static uint8_t leads = 0xFF;
static float batt_v = 0.0f;
static float batt_pct = 0.0f;
static float temp = 0.0f;

static uint16_t pkt_seq = 0;
static uint32_t ecg_cnt = 0;
static uint32_t ppg_cnt = 0;
static uint32_t pkt_cnt = 0;

static esp_gatt_if_t gatt_if = 0;
static uint16_t conn_id = 0;
static uint16_t char_handle = 0;
static bool connected = false;
static bool notify_en = false;

// ============================================================================
// CRC
// ============================================================================

static uint16_t crc16(uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : crc >> 1;
        }
    }
    return crc;
}

// ============================================================================
// HARDWARE INIT
// ============================================================================

static void init_i2c(i2c_port_t port, int sda, int scl) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = sda,
        .scl_io_num = scl,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = 400000,
    };
    i2c_param_config(port, &conf);
    i2c_driver_install(port, I2C_MODE_MASTER, 0, 0, 0);
}

static void i2c_write(i2c_port_t port, uint8_t addr, uint8_t reg, uint8_t val) {
    uint8_t buf[2] = {reg, val};
    i2c_master_write_to_device(port, addr, buf, 2, pdMS_TO_TICKS(10));
}

static esp_err_t i2c_read(i2c_port_t port, uint8_t addr, uint8_t reg, uint8_t *data, size_t len) {
    return i2c_master_write_read_device(port, addr, &reg, 1, data, len, pdMS_TO_TICKS(10));
}

static void init_max30101(void) {
    // Reset
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x09, 0x40);
    vTaskDelay(pdMS_TO_TICKS(50));
    
    // Configure for 50 Hz
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x04, 0x00); // FIFO clear
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x05, 0x00);
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x06, 0x00);
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x08, 0x0F); // Sample averaging: 1
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x09, 0x03); // Mode: SpO2
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x0A, 0x27); // 100 Hz, 411µs pulsewidth
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x0C, 0x24); // LED current
    i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x0D, 0x24);
    
    ESP_LOGI(TAG, "MAX30101: 50Hz mode");
}

// ============================================================================
// MASTER ACQUISITION TASK (CORE 1)
// ============================================================================

static void IRAM_ATTR master_timer_cb(void *arg) {
    BaseType_t xHigher = pdFALSE;
    vTaskNotifyGiveFromISR(acq_task_h, &xHigher);
    if (xHigher) portYIELD_FROM_ISR();
}

static void acquisition_task(void *arg) {
    packet_t pkt;
    uint8_t cycle = 0;
    uint8_t ppg_idx = 0;
    uint8_t fifo[6];
    
    memset(&pkt, 0, sizeof(pkt));
    
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        // EVERY CYCLE (125 Hz): Read ECG
        int adc_val = 0;
        adc_oneshot_read(adc1, ECG_ADC_CH, &adc_val);
        pkt.ecg[cycle] = (int16_t)adc_val;
        ecg_cnt++;
        
        // EVERY 2.5 CYCLES (~50 Hz): Read PPG
        if (cycle == 0 || cycle == 2) {
            if (i2c_read(I2C_PORT_PPG, MAX30101_ADDR, 0x07, fifo, 6) == ESP_OK) {
                uint32_t red = ((uint32_t)fifo[0] << 16) | ((uint32_t)fifo[1] << 8) | fifo[2];
                uint32_t ir = ((uint32_t)fifo[3] << 16) | ((uint32_t)fifo[4] << 8) | fifo[5];
                
                pkt.red[ppg_idx] = red & 0x3FFFF;
                pkt.ir[ppg_idx] = ir & 0x3FFFF;
                ppg_idx++;
                ppg_cnt++;
            }
        }
        
        // CYCLE 0: Read aux sensors + lead status
        if (cycle == 0) {
            // Lead status
            leads = 0;
            if (gpio_get_level(LO_PLUS)) leads |= 0x01;
            if (gpio_get_level(LO_MINUS)) leads |= 0x02;
            
            // Battery (fast read)
            uint8_t buf[2];
            if (i2c_read(I2C_PORT_FG, MAX17048_ADDR, 0x02, buf, 2) == ESP_OK) {
                uint16_t vcell = (buf[0] << 8) | buf[1];
                batt_v = (vcell >> 4) * 1.25f / 1000.0f;
            }
            if (i2c_read(I2C_PORT_FG, MAX17048_ADDR, 0x04, buf, 2) == ESP_OK) {
                uint16_t soc = (buf[0] << 8) | buf[1];
                batt_pct = (soc >> 8) + (soc & 0xFF) / 256.0f;
            }
            
            // Temperature (trigger)
            static uint8_t temp_state = 0;
            if (temp_state == 0) {
                i2c_write(I2C_PORT_PPG, MAX30101_ADDR, 0x21, 0x01);
                temp_state = 1;
            } else if (temp_state == 5) {
                uint8_t ti, tf;
                if (i2c_read(I2C_PORT_PPG, MAX30101_ADDR, 0x1F, &ti, 1) == ESP_OK &&
                    i2c_read(I2C_PORT_PPG, MAX30101_ADDR, 0x20, &tf, 1) == ESP_OK) {
                    temp = (int8_t)ti + (tf * 0.0625f);
                }
                temp_state = 0;
            } else {
                temp_state++;
            }
        }
        
        // CYCLE 4: Packet complete (5 ECG, 2 PPG)
        if (++cycle >= 5) {
            cycle = 0;
            ppg_idx = 0;
            
            pkt.timestamp = (uint32_t)(esp_timer_get_time() / 1000);
            pkt.seq = pkt_seq++;
            pkt.leads = leads;
            pkt.battery_v = batt_v;
            pkt.battery_pct = batt_pct;
            pkt.temp = temp;
            pkt.crc = crc16((uint8_t*)&pkt, sizeof(pkt) - 2);
            
            if (xQueueSend(packet_queue, &pkt, 0) == pdTRUE) {
                pkt_cnt++;
            }
            
            memset(&pkt, 0, sizeof(pkt));
        }
    }
}

// ============================================================================
// BLE TASK (CORE 0)
// ============================================================================

static void ble_task(void *arg) {
    packet_t pkt;
    TickType_t last = xTaskGetTickCount();
    
    while (1) {
        vTaskDelayUntil(&last, pdMS_TO_TICKS(40)); // 25 Hz
        
        if (xQueueReceive(packet_queue, &pkt, 0) == pdTRUE) {
            if (connected && notify_en && gatt_if != 0) {
                esp_ble_gatts_send_indicate(gatt_if, conn_id, char_handle,
                                           sizeof(pkt), (uint8_t*)&pkt, false);
            }
        }
    }
}

// ============================================================================
// MONITOR TASK
// ============================================================================

static void monitor_task(void *arg) {
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        uxTaskGetStackHighWaterMark(acq_task_h);
        ESP_LOGI(TAG, "ECG=%lu PPG=%lu PKT=%lu Q=%d Heap=%lu",
                 ecg_cnt, ppg_cnt, pkt_cnt, 
                 uxQueueMessagesWaiting(packet_queue),
                 esp_get_free_heap_size());
    }
}

// ============================================================================
// BLE HANDLERS
// ============================================================================

static void gap_handler(esp_gap_ble_cb_event_t event, esp_ble_gap_cb_param_t *param) {
    if (event == ESP_GAP_BLE_ADV_DATA_SET_COMPLETE_EVT) {
        esp_ble_gap_start_advertising(&(esp_ble_adv_params_t){
            .adv_int_min = 0x20, .adv_int_max = 0x40,
            .adv_type = ADV_TYPE_IND, .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
            .channel_map = ADV_CHNL_ALL, .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
        });
    }
}

static void gatts_handler(esp_gatts_cb_event_t event, esp_gatt_if_t gatts_if, 
                         esp_ble_gatts_cb_param_t *param) {
    switch (event) {
        case ESP_GATTS_REG_EVT:
            gatt_if = gatts_if;
            esp_ble_gap_set_device_name(DEVICE_NAME);
            esp_ble_gap_config_adv_data(&(esp_ble_adv_data_t){
                .set_scan_rsp = false, .include_name = true,
                .include_txpower = true,
                .flag = ESP_BLE_ADV_FLAG_GEN_DISC | ESP_BLE_ADV_FLAG_BREDR_NOT_SPT,
            });
            esp_ble_gatts_create_service(gatts_if, &(esp_gatt_srvc_id_t){
                .is_primary = true, .id.inst_id = 0,
                .id.uuid = {.len = ESP_UUID_LEN_16, .uuid.uuid16 = 0x180D},
            }, 4);
            break;
            
        case ESP_GATTS_CREATE_EVT:
            esp_ble_gatts_start_service(param->create.service_handle);
            esp_ble_gatts_add_char(param->create.service_handle, &(esp_bt_uuid_t){
                .len = ESP_UUID_LEN_16, .uuid.uuid16 = 0x2A37,
            }, ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE,
            ESP_GATT_CHAR_PROP_BIT_NOTIFY, NULL, NULL);
            break;
            
        case ESP_GATTS_ADD_CHAR_EVT:
            char_handle = param->add_char.attr_handle;
            esp_ble_gatts_add_char_descr(param->add_char.service_handle, &(esp_bt_uuid_t){
                .len = ESP_UUID_LEN_16, .uuid.uuid16 = ESP_GATT_UUID_CHAR_CLIENT_CONFIG,
            }, ESP_GATT_PERM_READ | ESP_GATT_PERM_WRITE, NULL, NULL);
            break;
            
        case ESP_GATTS_CONNECT_EVT:
            connected = true;
            conn_id = param->connect.conn_id;
            ESP_LOGI(TAG, "BLE connected");
            break;
            
        case ESP_GATTS_DISCONNECT_EVT:
            connected = false;
            notify_en = false;
            esp_ble_gap_start_advertising(&(esp_ble_adv_params_t){
                .adv_int_min = 0x20, .adv_int_max = 0x40,
                .adv_type = ADV_TYPE_IND, .own_addr_type = BLE_ADDR_TYPE_PUBLIC,
                .channel_map = ADV_CHNL_ALL, .adv_filter_policy = ADV_FILTER_ALLOW_SCAN_ANY_CON_ANY,
            });
            ESP_LOGI(TAG, "BLE disconnected");
            break;
            
        case ESP_GATTS_WRITE_EVT:
            if (param->write.len == 2) {
                uint16_t val = param->write.value[0] | (param->write.value[1] << 8);
                notify_en = (val == 0x0001);
                ESP_LOGI(TAG, "Notify %s", notify_en ? "ON" : "OFF");
            }
            if (param->write.need_rsp) {
                esp_ble_gatts_send_response(gatts_if, param->write.conn_id,
                                           param->write.trans_id, ESP_GATT_OK, NULL);
            }
            break;
            
        case ESP_GATTS_MTU_EVT:
            ESP_LOGI(TAG, "MTU=%d", param->mtu.mtu);
            break;
            
        default:
            break;
    }
}

// ============================================================================
// MAIN
// ============================================================================

void app_main(void) {
    ESP_LOGI(TAG, "=== Nirog Optimized v1.0 ===");
    ESP_LOGI(TAG, "ECG=%dHz PPG=%dHz PKT=%dHz", ECG_RATE_HZ, PPG_RATE_HZ, PACKET_RATE_HZ);
    
    nvs_flash_init();
    
    gpio_config(&(gpio_config_t){
        .pin_bit_mask = (1ULL << LO_PLUS) | (1ULL << LO_MINUS),
        .mode = GPIO_MODE_INPUT,
    });
    
    adc_oneshot_new_unit(&(adc_oneshot_unit_init_cfg_t){.unit_id = ADC_UNIT_1}, &adc1);
    adc_oneshot_config_channel(adc1, ECG_ADC_CH, &(adc_oneshot_chan_cfg_t){
        .bitwidth = ADC_BITWIDTH_12, .atten = ADC_ATTEN_DB_12,
    });
    
    init_i2c(I2C_PORT_PPG, I2C_SDA_PPG, I2C_SCL_PPG);
    init_i2c(I2C_PORT_FG, I2C_SDA_FG, I2C_SCL_FG);
    init_max30101();
    
    packet_queue = xQueueCreate(PACKET_QUEUE_SIZE, sizeof(packet_t));
    
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    esp_bt_controller_init(&bt_cfg);
    esp_bt_controller_enable(ESP_BT_MODE_BLE);
    esp_bluedroid_init();
    esp_bluedroid_enable();
    esp_ble_gatts_register_callback(gatts_handler);
    esp_ble_gap_register_callback(gap_handler);
    esp_ble_gatts_app_register(0);
    
    xTaskCreatePinnedToCore(acquisition_task, "acq", 4096, NULL, 5, &acq_task_h, 1);
    xTaskCreatePinnedToCore(ble_task, "ble", 4096, NULL, 6, &ble_task_h, 0);
    xTaskCreatePinnedToCore(monitor_task, "mon", 4096, NULL, 1, NULL, 0);
    
    esp_timer_create(&(esp_timer_create_args_t){.callback = master_timer_cb, .name = "master"}, &master_timer);
    esp_timer_start_periodic(master_timer, MASTER_PERIOD_US);
    
    ESP_LOGI(TAG, "Running");
}