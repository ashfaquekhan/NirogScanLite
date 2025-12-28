#include <stdio.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "nvs_flash.h"
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"
#include "esp_gatt_common_api.h"
#include "esp_bt_main.h"

#define TAG "Nirog"

#define ECG_RATE_HZ          125
#define PPG_RATE_HZ          25
#define PACKET_RATE_HZ       25

#define MASTER_RATE_HZ       125
#define MASTER_PERIOD_US     (1000000 / MASTER_RATE_HZ)

#define ECG_ADC_CH           ADC_CHANNEL_1
#define LO_PLUS              GPIO_NUM_16
#define LO_MINUS             GPIO_NUM_17
#define I2C_SCL_PPG          GPIO_NUM_9
#define I2C_SDA_PPG          GPIO_NUM_8
#define I2C_SCL_FG           GPIO_NUM_11
#define I2C_SDA_FG           GPIO_NUM_10

#define MAX30101_ADDR        0x57
#define MAX17048_ADDR        0x36

#define DEVICE_NAME          "NirogScan"
#define PACKET_QUEUE_SIZE    8

#define MAX3010X_REG_FIFO_WRITE_PTR     0x04
#define MAX3010X_REG_FIFO_OVERFLOW      0x05
#define MAX3010X_REG_FIFO_READ_PTR      0x06
#define MAX3010X_REG_FIFO_DATA          0x07
#define MAX3010X_REG_FIFO_CONFIG        0x08
#define MAX3010X_REG_MODE_CONFIG        0x09
#define MAX3010X_REG_SPO2_CONFIG        0x0A
#define MAX3010X_REG_LED1_PA            0x0C
#define MAX3010X_REG_LED2_PA            0x0D
#define MAX3010X_REG_TEMP_INT           0x1F
#define MAX3010X_REG_TEMP_FRAC          0x20
#define MAX3010X_REG_TEMP_CONFIG        0x21
#define MAX3010X_REG_PART_ID            0xFF

#define MAX3010X_MODE_RESET             0x40
#define MAX3010X_MODE_SPO2              0x03
#define MAX3010X_FIFO_SMP_AVE_4         0x40
#define MAX3010X_FIFO_ROLLOVER_EN       0x10
#define MAX3010X_FIFO_A_FULL_17         0x0F
#define MAX3010X_SPO2_ADC_RGE_4096      0x20
#define MAX3010X_SPO2_SR_100            0x04
#define MAX3010X_SPO2_PW_411US          0x03

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

static adc_oneshot_unit_handle_t adc1 = NULL;
static esp_timer_handle_t master_timer = NULL;
static TaskHandle_t acq_task_h = NULL;
static TaskHandle_t ble_task_h = NULL;
static QueueHandle_t packet_queue = NULL;

static i2c_master_bus_handle_t i2c_bus_ppg = NULL;
static i2c_master_bus_handle_t i2c_bus_fg = NULL;
static i2c_master_dev_handle_t max30101_dev = NULL;
static i2c_master_dev_handle_t max17048_dev = NULL;

static volatile uint8_t leads = 0xFF;
static volatile float batt_v = 0.0f;
static volatile float batt_pct = 0.0f;
static volatile float temp_c = 0.0f;

static uint16_t pkt_seq = 0;
static uint32_t ecg_cnt = 0;
static uint32_t ppg_cnt = 0;
static uint32_t pkt_cnt = 0;
static uint32_t fifo_empty_cnt = 0;
static uint32_t fifo_ovf_cnt = 0;
static uint32_t queue_full_cnt = 0;
static uint32_t ble_send_cnt = 0;

static volatile esp_gatt_if_t gatt_if = 0;
static volatile uint16_t conn_id = 0;
static volatile uint16_t char_handle = 0;
static volatile bool connected = false;
static volatile bool notify_en = false;
static volatile uint16_t current_mtu = 23;

static uint8_t temp_state = 0;

static uint32_t last_ppg_red = 0;
static uint32_t last_ppg_ir = 0;
static bool ppg_valid = false;

static uint16_t crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFF;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc & 1) ? (crc >> 1) ^ 0xA001 : crc >> 1;
        }
    }
    return crc;
}

static esp_err_t init_i2c_ppg(void) {
    i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = I2C_SDA_PPG,
        .scl_io_num = I2C_SCL_PPG,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    
    esp_err_t ret = i2c_new_master_bus(&bus_config, &i2c_bus_ppg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "PPG I2C bus failed: %s", esp_err_to_name(ret));
        return ret;
    }
    
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MAX30101_ADDR,
        .scl_speed_hz = 400000,
    };
    
    ret = i2c_master_bus_add_device(i2c_bus_ppg, &dev_cfg, &max30101_dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MAX30101 add failed: %s", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t init_i2c_fg(void) {
    i2c_master_bus_config_t bus_config = {
        .i2c_port = I2C_NUM_1,
        .sda_io_num = I2C_SDA_FG,
        .scl_io_num = I2C_SCL_FG,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = false,
    };
    
    esp_err_t ret = i2c_new_master_bus(&bus_config, &i2c_bus_fg);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "FG I2C bus failed: %s", esp_err_to_name(ret));
        return ret;
    }
    
    i2c_device_config_t dev_cfg = {
        .dev_addr_length = I2C_ADDR_BIT_LEN_7,
        .device_address = MAX17048_ADDR,
        .scl_speed_hz = 100000,
    };
    
    ret = i2c_master_bus_add_device(i2c_bus_fg, &dev_cfg, &max17048_dev);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MAX17048 add failed: %s", esp_err_to_name(ret));
    }
    return ret;
}

static esp_err_t i2c_write_reg(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t val) {
    uint8_t buf[2] = {reg, val};
    return i2c_master_transmit(dev, buf, 2, 100);
}

static esp_err_t i2c_read_reg(i2c_master_dev_handle_t dev, uint8_t reg, uint8_t *data, size_t len) {
    return i2c_master_transmit_receive(dev, &reg, 1, data, len, 100);
}

static bool max30101_read_fifo_sample(uint32_t *red, uint32_t *ir) {
    uint8_t wr_ptr = 0, rd_ptr = 0, ovf = 0;
    
    if (i2c_read_reg(max30101_dev, MAX3010X_REG_FIFO_OVERFLOW, &ovf, 1) == ESP_OK && ovf > 0) {
        fifo_ovf_cnt += ovf;
        i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_WRITE_PTR, 0x00);
        i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_OVERFLOW, 0x00);
        i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_READ_PTR, 0x00);
        return false;
    }
    
    if (i2c_read_reg(max30101_dev, MAX3010X_REG_FIFO_WRITE_PTR, &wr_ptr, 1) != ESP_OK) return false;
    if (i2c_read_reg(max30101_dev, MAX3010X_REG_FIFO_READ_PTR, &rd_ptr, 1) != ESP_OK) return false;
    
    if (wr_ptr == rd_ptr) {
        fifo_empty_cnt++;
        return false;
    }
    
    uint8_t fifo[6];
    if (i2c_read_reg(max30101_dev, MAX3010X_REG_FIFO_DATA, fifo, 6) != ESP_OK) {
        return false;
    }
    
    *red = (((uint32_t)fifo[0] << 16) | ((uint32_t)fifo[1] << 8) | fifo[2]) & 0x3FFFF;
    *ir = (((uint32_t)fifo[3] << 16) | ((uint32_t)fifo[4] << 8) | fifo[5]) & 0x3FFFF;
    
    return true;
}

static void init_max30101(void) {
    esp_err_t ret;
    
    ret = i2c_write_reg(max30101_dev, MAX3010X_REG_MODE_CONFIG, MAX3010X_MODE_RESET);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "MAX30101 reset failed: %s", esp_err_to_name(ret));
        return;
    }
    vTaskDelay(pdMS_TO_TICKS(100));
    
    i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_WRITE_PTR, 0x00);
    i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_OVERFLOW, 0x00);
    i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_READ_PTR, 0x00);
    
    uint8_t fifo_cfg = MAX3010X_FIFO_SMP_AVE_4 | MAX3010X_FIFO_ROLLOVER_EN | MAX3010X_FIFO_A_FULL_17;
    i2c_write_reg(max30101_dev, MAX3010X_REG_FIFO_CONFIG, fifo_cfg);
    
    i2c_write_reg(max30101_dev, MAX3010X_REG_MODE_CONFIG, MAX3010X_MODE_SPO2);
    
    uint8_t spo2_cfg = MAX3010X_SPO2_ADC_RGE_4096 | MAX3010X_SPO2_SR_100 | MAX3010X_SPO2_PW_411US;
    i2c_write_reg(max30101_dev, MAX3010X_REG_SPO2_CONFIG, spo2_cfg);
    
    i2c_write_reg(max30101_dev, MAX3010X_REG_LED1_PA, 0x1F);
    i2c_write_reg(max30101_dev, MAX3010X_REG_LED2_PA, 0x1F);
    
    uint8_t part_id = 0;
    ret = i2c_read_reg(max30101_dev, MAX3010X_REG_PART_ID, &part_id, 1);
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "MAX30101: ID=0x%02X [100Hz/4xAvg=25Hz, 411us, 4096nA, 6.2mA]", part_id);
    } else {
        ESP_LOGE(TAG, "MAX30101: Read failed");
    }
}

static void IRAM_ATTR master_timer_cb(void *arg) {
    BaseType_t xHigher = pdFALSE;
    vTaskNotifyGiveFromISR(acq_task_h, &xHigher);
    if (xHigher) portYIELD_FROM_ISR();
}

static void acquisition_task(void *arg) {
    packet_t pkt;
    uint8_t cycle = 0;
    
    vTaskDelay(pdMS_TO_TICKS(100));
    
    memset(&pkt, 0, sizeof(pkt));
    
    while (1) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        
        int adc_val = 0;
        adc_oneshot_read(adc1, ECG_ADC_CH, &adc_val);
        pkt.ecg[cycle] = (int16_t)adc_val;
        ecg_cnt++;
        
        if (cycle == 0) {
            uint32_t red = 0, ir = 0;
            if (max30101_read_fifo_sample(&red, &ir)) {
                last_ppg_red = red;
                last_ppg_ir = ir;
                ppg_valid = true;
                ppg_cnt++;
            }
            
            if (ppg_valid) {
                pkt.red[0] = last_ppg_red;
                pkt.ir[0] = last_ppg_ir;
            } else {
                pkt.red[0] = 0;
                pkt.ir[0] = 0;
            }
            pkt.red[1] = 0;
            pkt.ir[1] = 0;
            
            leads = 0;
            if (gpio_get_level(LO_PLUS)) leads |= 0x01;
            if (gpio_get_level(LO_MINUS)) leads |= 0x02;
        }
        
        if (cycle == 2) {
            uint8_t buf[2];
            if (i2c_read_reg(max17048_dev, 0x02, buf, 2) == ESP_OK) {
                uint16_t vcell = (buf[0] << 8) | buf[1];
                batt_v = (vcell >> 4) * 1.25f / 1000.0f;
            }
            if (i2c_read_reg(max17048_dev, 0x04, buf, 2) == ESP_OK) {
                uint16_t soc = (buf[0] << 8) | buf[1];
                batt_pct = (soc >> 8) + (soc & 0xFF) / 256.0f;
            }
        }
        
        if (cycle == 3) {
            if (temp_state == 0) {
                i2c_write_reg(max30101_dev, MAX3010X_REG_TEMP_CONFIG, 0x01);
                temp_state = 1;
            } else if (temp_state >= 5) {
                uint8_t ti = 0, tf = 0;
                if (i2c_read_reg(max30101_dev, MAX3010X_REG_TEMP_INT, &ti, 1) == ESP_OK &&
                    i2c_read_reg(max30101_dev, MAX3010X_REG_TEMP_FRAC, &tf, 1) == ESP_OK) {
                    temp_c = (int8_t)ti + (tf * 0.0625f);
                }
                temp_state = 0;
            } else {
                temp_state++;
            }
        }
        
        if (++cycle >= 5) {
            cycle = 0;
            
            pkt.timestamp = (uint32_t)(esp_timer_get_time() / 1000);
            pkt.seq = pkt_seq++;
            pkt.leads = leads;
            pkt.battery_v = batt_v;
            pkt.battery_pct = batt_pct;
            pkt.temp = temp_c;
            pkt.crc = crc16((const uint8_t*)&pkt, sizeof(pkt) - 2);
            
            if (xQueueSend(packet_queue, &pkt, 0) == pdTRUE) {
                pkt_cnt++;
            } else {
                queue_full_cnt++;
            }
            
            memset(&pkt, 0, sizeof(pkt));
        }
    }
}

static void ble_task(void *arg) {
    packet_t pkt;
    TickType_t last_wake = xTaskGetTickCount();
    
    while (1) {
        vTaskDelayUntil(&last_wake, pdMS_TO_TICKS(40));
        
        bool is_connected = connected;
        bool is_notify = notify_en;
        esp_gatt_if_t cur_gatt_if = gatt_if;
        uint16_t cur_conn_id = conn_id;
        uint16_t cur_char_handle = char_handle;
        uint16_t cur_mtu = current_mtu;
        
        while (xQueueReceive(packet_queue, &pkt, 0) == pdTRUE) {
            if (is_connected && is_notify && cur_gatt_if != 0) {
                if (cur_mtu >= sizeof(pkt) + 3) {
                    esp_err_t ret = esp_ble_gatts_send_indicate(cur_gatt_if, cur_conn_id, cur_char_handle,
                                                                sizeof(pkt), (uint8_t*)&pkt, false);
                    if (ret == ESP_OK) {
                        ble_send_cnt++;
                    }
                }
            }
        }
    }
}

static void monitor_task(void *arg) {
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(30000));
        
        ESP_LOGI(TAG, "ECG=%lu PPG=%lu PKT=%lu", ecg_cnt, ppg_cnt, pkt_cnt);
        ESP_LOGI(TAG, "FIFO_EMPTY=%lu OVF=%lu Q_FULL=%lu BLE_TX=%lu MTU=%d", 
                 fifo_empty_cnt, fifo_ovf_cnt, queue_full_cnt, ble_send_cnt, current_mtu);
    }
}

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
                .set_scan_rsp = false, .include_name = true, .include_txpower = true,
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
            conn_id = param->connect.conn_id;
            current_mtu = 23;
            connected = true;
            ESP_LOGI(TAG, "BLE connected");
            break;
            
        case ESP_GATTS_DISCONNECT_EVT:
            connected = false;
            notify_en = false;
            current_mtu = 23;
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
            current_mtu = param->mtu.mtu;
            ESP_LOGI(TAG, "MTU=%d", current_mtu);
            break;
            
        default:
            break;
    }
}

void app_main(void) {
    ESP_LOGI(TAG, "=== Nirog v2.6 ===");
    ESP_LOGI(TAG, "ECG=%dHz PPG=%dHz PKT=%dHz SIZE=%d", 
             ECG_RATE_HZ, PPG_RATE_HZ, PACKET_RATE_HZ, sizeof(packet_t));
    
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        nvs_flash_init();
    }
    
    gpio_config(&(gpio_config_t){
        .pin_bit_mask = (1ULL << LO_PLUS) | (1ULL << LO_MINUS),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    });
    
    adc_oneshot_new_unit(&(adc_oneshot_unit_init_cfg_t){.unit_id = ADC_UNIT_1}, &adc1);
    adc_oneshot_config_channel(adc1, ECG_ADC_CH, &(adc_oneshot_chan_cfg_t){
        .bitwidth = ADC_BITWIDTH_12, .atten = ADC_ATTEN_DB_12,
    });
    
    if (init_i2c_ppg() != ESP_OK) {
        ESP_LOGE(TAG, "PPG I2C failed");
    }
    if (init_i2c_fg() != ESP_OK) {
        ESP_LOGE(TAG, "FG I2C failed");
    }
    init_max30101();
    
    packet_queue = xQueueCreate(PACKET_QUEUE_SIZE, sizeof(packet_t));
    if (packet_queue == NULL) {
        ESP_LOGE(TAG, "Queue create failed");
        return;
    }
    
    esp_bt_controller_config_t bt_cfg = BT_CONTROLLER_INIT_CONFIG_DEFAULT();
    esp_bt_controller_init(&bt_cfg);
    esp_bt_controller_enable(ESP_BT_MODE_BLE);
    esp_bluedroid_init();
    esp_bluedroid_enable();
    esp_ble_gatt_set_local_mtu(185);
    esp_ble_gatts_register_callback(gatts_handler);
    esp_ble_gap_register_callback(gap_handler);
    esp_ble_gatts_app_register(0);
    
    xTaskCreatePinnedToCore(acquisition_task, "acq", 4096, NULL, 5, &acq_task_h, 1);
    xTaskCreatePinnedToCore(ble_task, "ble", 4096, NULL, 4, &ble_task_h, 0);
    xTaskCreatePinnedToCore(monitor_task, "mon", 4096, NULL, 1, NULL, 0);
    
    esp_timer_create(&(esp_timer_create_args_t){.callback = master_timer_cb, .name = "master"}, &master_timer);
    esp_timer_start_periodic(master_timer, MASTER_PERIOD_US);
    
    ESP_LOGI(TAG, "Running");
}