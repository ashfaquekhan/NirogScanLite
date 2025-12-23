/**
 * @file nirog_scan_optimized.c
 * @brief ESP32S3 Multi-Sensor Data Acquisition System
 * 
 * HARDWARE CONFIGURATION:
 * - MAX30101: PPG sensor (I2C: SDA=8, SCL=9)
 * - AD8232: ECG sensor (ADC1_CH1=GPIO2, LO+=GPIO16, LO-=GPIO17)
 * - MAX17048: Fuel gauge (I2C: SDA=8, SCL=9)
 * 
 * OUTPUT FORMATS:
 * 1. PACKET FORMAT (OUTPUT_FORMAT_PACKET):
 *    PKT>seq:123,ppg_red:45678,12345,ppg_ir:54321,23456,ecg:2048,2056,2044,2052,2048,lo:0,0,0,0,0,batt_v:3.756,batt_soc:78.5,temp:36.45,t_start:1234567890,t_end:1234567895
 * 
 * 2. APP FORMAT (OUTPUT_FORMAT_APP):
 *    >A1:45678,12345,A2:54321,23456,A3:2048,2056,2044,2052,2048,A4:0,0,0,0,0,A5:3.756,A6:78.5,A7:36.45,A8:123,A9:1234567890,A10:1234567895
 *    Where: A1=PPG_Red, A2=PPG_IR, A3=ECG, A4=LO_Status, A5=Battery_V, A6=Battery_SOC, A7=Temp, A8=Sequence, A9=StartTime, A10=EndTime
 * 
 * TIMING CONFIGURATION:
 * - PPG: 50Hz (2 samples per packet)
 * - ECG: 125Hz (5 samples per packet)  
 * - Packet Rate: 25Hz (40ms intervals)
 * 
 * TO CHANGE OUTPUT FORMAT: Modify OUTPUT_FORMAT define below
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_err.h"
#include "esp_check.h"

static const char *TAG = "NirogScan";

#define I2C_MASTER_SCL_IO           9
#define I2C_MASTER_SDA_IO           8
#define I2C_MASTER_FREQ_HZ          400000
#define I2C_MASTER_NUM              I2C_NUM_0
#define I2C_MASTER_TIMEOUT_MS       100

#define MAX30101_I2C_ADDR           0x57
#define MAX17048_I2C_ADDR           0x36

#define ECG_ADC_CHANNEL             ADC_CHANNEL_1
#define ECG_ADC_ATTEN               ADC_ATTEN_DB_12
#define ECG_ADC_BITWIDTH            ADC_BITWIDTH_12
#define ECG_GPIO_NUM                GPIO_NUM_2
#define LO_PLUS_PIN                 GPIO_NUM_16
#define LO_MINUS_PIN                GPIO_NUM_17

#define MAX30101_REG_FIFO_WR_PTR    0x04
#define MAX30101_REG_FIFO_RD_PTR    0x06
#define MAX30101_REG_FIFO_DATA      0x07
#define MAX30101_REG_FIFO_CONFIG    0x08
#define MAX30101_REG_MODE_CONFIG    0x09
#define MAX30101_REG_SPO2_CONFIG    0x0A
#define MAX30101_REG_LED1_PA        0x0C
#define MAX30101_REG_LED2_PA        0x0D
#define MAX30101_REG_TEMP_INT       0x1F
#define MAX30101_REG_TEMP_FRAC      0x20
#define MAX30101_REG_TEMP_CONFIG    0x21
#define MAX30101_REG_PART_ID        0xFF

#define MAX17048_REG_VCELL          0x02
#define MAX17048_REG_SOC            0x04

#define PPG_SAMPLE_RATE_HZ          50
#define ECG_SAMPLE_RATE_HZ          125
#define PPG_TIMER_PERIOD_US         20000
#define ECG_TIMER_PERIOD_US         8000
#define PACKET_TIMER_PERIOD_US      40000

#define PPG_SAMPLES_PER_PACKET      2
#define ECG_SAMPLES_PER_PACKET      5

#define QUEUE_SIZE                  16

#define OUTPUT_FORMAT_PACKET        0
#define OUTPUT_FORMAT_APP           1
#define OUTPUT_FORMAT               OUTPUT_FORMAT_APP

typedef struct {
    uint32_t red;
    uint32_t ir;
    uint64_t timestamp_us;
} ppg_sample_t;

typedef struct {
    uint16_t ecg_value;
    uint8_t lo_status;
    uint64_t timestamp_us;
} ecg_sample_t;

typedef struct {
    uint32_t ppg_red[PPG_SAMPLES_PER_PACKET];
    uint32_t ppg_ir[PPG_SAMPLES_PER_PACKET];
    uint16_t ecg_values[ECG_SAMPLES_PER_PACKET];
    uint8_t lo_status[ECG_SAMPLES_PER_PACKET];
    float battery_voltage;
    float battery_soc;
    float temperature;
    uint64_t packet_start_timestamp;
    uint64_t packet_end_timestamp;
    uint32_t packet_sequence;
} unified_packet_t;

static adc_oneshot_unit_handle_t adc1_handle = NULL;
static esp_timer_handle_t ppg_timer = NULL;
static esp_timer_handle_t ecg_timer = NULL;
static esp_timer_handle_t packet_timer = NULL;

static QueueHandle_t ppg_queue = NULL;
static QueueHandle_t ecg_queue = NULL;
static QueueHandle_t packet_queue = NULL;

static SemaphoreHandle_t system_data_mutex = NULL;

static struct {
    float battery_voltage;
    float battery_soc;
    float temperature;
    bool temperature_valid;
    uint32_t temp_read_counter;
} system_data = {0};

static esp_err_t i2c_master_init(void)
{
    const i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ,
    };
    
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM, &conf), TAG, "I2C param config failed");
    return i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
}

static esp_err_t i2c_write_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t data)
{
    const uint8_t write_buf[2] = {reg_addr, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM, dev_addr, write_buf, 2, 
                                     pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

static esp_err_t i2c_read_reg(uint8_t dev_addr, uint8_t reg_addr, uint8_t *data, size_t len)
{
    return i2c_master_write_read_device(I2C_MASTER_NUM, dev_addr, &reg_addr, 1, 
                                       data, len, pdMS_TO_TICKS(I2C_MASTER_TIMEOUT_MS));
}

static esp_err_t max30101_init(void)
{
    uint8_t part_id;
    ESP_RETURN_ON_ERROR(i2c_read_reg(MAX30101_I2C_ADDR, MAX30101_REG_PART_ID, &part_id, 1), 
                        TAG, "Failed to read part ID");
    
    if (part_id != 0x15) {
        ESP_LOGE(TAG, "Invalid part ID: 0x%02X", part_id);
        return ESP_ERR_NOT_FOUND;
    }
    
    ESP_LOGI(TAG, "MAX30101 Part ID: 0x%02X", part_id);

    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_MODE_CONFIG, 0x40), 
                        TAG, "Reset failed");
    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_FIFO_WR_PTR, 0x00), 
                        TAG, "FIFO WR PTR failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_FIFO_RD_PTR, 0x00), 
                        TAG, "FIFO RD PTR failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_FIFO_CONFIG, 0x4F), 
                        TAG, "FIFO config failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_MODE_CONFIG, 0x03), 
                        TAG, "Mode config failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_SPO2_CONFIG, 0x03), 
                        TAG, "SPO2 config failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_LED1_PA, 0x24), 
                        TAG, "LED1 PA failed");
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_LED2_PA, 0x24), 
                        TAG, "LED2 PA failed");

    return ESP_OK;
}

static esp_err_t max30101_read_fifo(uint32_t *red, uint32_t *ir)
{
    uint8_t fifo_data[6];
    ESP_RETURN_ON_ERROR(i2c_read_reg(MAX30101_I2C_ADDR, MAX30101_REG_FIFO_DATA, fifo_data, 6), 
                        TAG, "FIFO read failed");
    
    *red = ((uint32_t)fifo_data[0] << 16) | ((uint32_t)fifo_data[1] << 8) | fifo_data[2];
    *red &= 0x3FFFF;
    
    *ir = ((uint32_t)fifo_data[3] << 16) | ((uint32_t)fifo_data[4] << 8) | fifo_data[5];
    *ir &= 0x3FFFF;
    
    return ESP_OK;
}

static esp_err_t max30101_read_temperature(float *temperature)
{
    ESP_RETURN_ON_ERROR(i2c_write_reg(MAX30101_I2C_ADDR, MAX30101_REG_TEMP_CONFIG, 0x01), 
                        TAG, "Temp config failed");
    
    vTaskDelay(pdMS_TO_TICKS(100));
    
    uint8_t temp_int, temp_frac;
    ESP_RETURN_ON_ERROR(i2c_read_reg(MAX30101_I2C_ADDR, MAX30101_REG_TEMP_INT, &temp_int, 1), 
                        TAG, "Temp int read failed");
    ESP_RETURN_ON_ERROR(i2c_read_reg(MAX30101_I2C_ADDR, MAX30101_REG_TEMP_FRAC, &temp_frac, 1), 
                        TAG, "Temp frac read failed");
    
    *temperature = (float)(int8_t)temp_int + ((float)temp_frac * 0.0625f);
    
    return ESP_OK;
}

static esp_err_t max17048_read_data(float *voltage, float *soc)
{
    uint8_t data[2];
    
    ESP_RETURN_ON_ERROR(i2c_read_reg(MAX17048_I2C_ADDR, MAX17048_REG_VCELL, data, 2), 
                        TAG, "VCELL read failed");
    uint16_t vcell = (data[0] << 8) | data[1];
    *voltage = (float)(vcell >> 4) * 1.25f / 1000.0f;
    
    ESP_RETURN_ON_ERROR(i2c_read_reg(MAX17048_I2C_ADDR, MAX17048_REG_SOC, data, 2), 
                        TAG, "SOC read failed");
    uint16_t soc_reg = (data[0] << 8) | data[1];
    *soc = (float)(soc_reg >> 8) + (float)(soc_reg & 0xFF) / 256.0f;
    
    return ESP_OK;
}

static esp_err_t adc_init(void)
{
    const adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_RETURN_ON_ERROR(adc_oneshot_new_unit(&init_config, &adc1_handle), 
                        TAG, "ADC unit init failed");

    const adc_oneshot_chan_cfg_t config = {
        .bitwidth = ECG_ADC_BITWIDTH,
        .atten = ECG_ADC_ATTEN,
    };
    return adc_oneshot_config_channel(adc1_handle, ECG_ADC_CHANNEL, &config);
}

static void gpio_init_all(void)
{
    const gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LO_PLUS_PIN) | (1ULL << LO_MINUS_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
}

static void IRAM_ATTR ppg_timer_callback(void* arg)
{
    BaseType_t higher_priority_task_woken = pdFALSE;
    ppg_sample_t sample;
    
    uint32_t red, ir;
    if (max30101_read_fifo(&red, &ir) == ESP_OK && red > 1000 && ir > 1000) {
        sample.red = red;
        sample.ir = ir;
        sample.timestamp_us = esp_timer_get_time();
        
        xQueueSendFromISR(ppg_queue, &sample, &higher_priority_task_woken);
    }
    
    if (higher_priority_task_woken) {
        portYIELD_FROM_ISR();
    }
}

static void IRAM_ATTR ecg_timer_callback(void* arg)
{
    BaseType_t higher_priority_task_woken = pdFALSE;
    ecg_sample_t sample;
    
    int adc_raw;
    if (adc_oneshot_read(adc1_handle, ECG_ADC_CHANNEL, &adc_raw) == ESP_OK) {
        sample.ecg_value = (uint16_t)adc_raw;
        sample.lo_status = 0;
        sample.lo_status |= gpio_get_level(LO_PLUS_PIN) ? 0x01 : 0x00;
        sample.lo_status |= gpio_get_level(LO_MINUS_PIN) ? 0x02 : 0x00;
        sample.timestamp_us = esp_timer_get_time();
        
        xQueueSendFromISR(ecg_queue, &sample, &higher_priority_task_woken);
        
        static uint32_t counter = 0;
        counter++;
        if (counter >= 125) {
            counter = 0;
            float voltage, soc, temperature;
            
            if (max17048_read_data(&voltage, &soc) == ESP_OK) {
                if (xSemaphoreTakeFromISR(system_data_mutex, &higher_priority_task_woken) == pdTRUE) {
                    system_data.battery_voltage = voltage;
                    system_data.battery_soc = soc;
                    
                    system_data.temp_read_counter++;
                    if (system_data.temp_read_counter >= 5) {
                        system_data.temp_read_counter = 0;
                        if (max30101_read_temperature(&temperature) == ESP_OK) {
                            system_data.temperature = temperature;
                            system_data.temperature_valid = true;
                        }
                    }
                    
                    xSemaphoreGiveFromISR(system_data_mutex, &higher_priority_task_woken);
                }
            }
        }
    }
    
    if (higher_priority_task_woken) {
        portYIELD_FROM_ISR();
    }
}

static void IRAM_ATTR packet_timer_callback(void* arg)
{
    BaseType_t higher_priority_task_woken = pdFALSE;
    const uint32_t signal = 1;
    xQueueSendFromISR(packet_queue, &signal, &higher_priority_task_woken);
    
    if (higher_priority_task_woken) {
        portYIELD_FROM_ISR();
    }
}

static void print_packet_format(const unified_packet_t *packet)
{
    printf("PKT>seq:%lu,ppg_red:", (unsigned long)packet->packet_sequence);
    for (int i = 0; i < PPG_SAMPLES_PER_PACKET; i++) {
        printf("%lu", (unsigned long)packet->ppg_red[i]);
        if (i < PPG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    printf(",ppg_ir:");
    for (int i = 0; i < PPG_SAMPLES_PER_PACKET; i++) {
        printf("%lu", (unsigned long)packet->ppg_ir[i]);
        if (i < PPG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    printf(",ecg:");
    for (int i = 0; i < ECG_SAMPLES_PER_PACKET; i++) {
        printf("%u", packet->ecg_values[i]);
        if (i < ECG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    printf(",lo:");
    for (int i = 0; i < ECG_SAMPLES_PER_PACKET; i++) {
        printf("%u", packet->lo_status[i]);
        if (i < ECG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    printf(",batt_v:%.3f,batt_soc:%.1f,temp:%.2f,t_start:%llu,t_end:%llu\n",
           packet->battery_voltage, packet->battery_soc, packet->temperature,
           packet->packet_start_timestamp, packet->packet_end_timestamp);
}

static void print_app_format(const unified_packet_t *packet)
{
    printf(">A1:");
    for (int i = 0; i < PPG_SAMPLES_PER_PACKET; i++) {
        printf("%lu", (unsigned long)packet->ppg_red[i]);
        if (i < PPG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    
    printf(",A2:");
    for (int i = 0; i < PPG_SAMPLES_PER_PACKET; i++) {
        printf("%lu", (unsigned long)packet->ppg_ir[i]);
        if (i < PPG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    
    printf(",A3:");
    for (int i = 0; i < ECG_SAMPLES_PER_PACKET; i++) {
        printf("%u", packet->ecg_values[i]);
        if (i < ECG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    
    printf(",A4:");
    for (int i = 0; i < ECG_SAMPLES_PER_PACKET; i++) {
        printf("%u", packet->lo_status[i]);
        if (i < ECG_SAMPLES_PER_PACKET - 1) printf(",");
    }
    
    printf(",A5:%.3f,A6:%.1f,A7:%.2f,A8:%lu,A9:%llu,A10:%llu\n",
           packet->battery_voltage, 
           packet->battery_soc, 
           packet->temperature,
           (unsigned long)packet->packet_sequence,
           packet->packet_start_timestamp, 
           packet->packet_end_timestamp);
}

static void packet_processing_task(void *pvParameters)
{
    unified_packet_t packet;
    uint32_t packet_sequence = 0;
    uint32_t signal;
    
    while (1) {
        if (xQueueReceive(packet_queue, &signal, portMAX_DELAY) == pdTRUE) {
            packet.packet_sequence = packet_sequence++;
            packet.packet_start_timestamp = esp_timer_get_time();
            
            for (int i = 0; i < PPG_SAMPLES_PER_PACKET; i++) {
                ppg_sample_t ppg_sample;
                if (xQueueReceive(ppg_queue, &ppg_sample, pdMS_TO_TICKS(5)) == pdTRUE) {
                    packet.ppg_red[i] = ppg_sample.red;
                    packet.ppg_ir[i] = ppg_sample.ir;
                } else {
                    packet.ppg_red[i] = (i > 0) ? packet.ppg_red[i-1] : 0;
                    packet.ppg_ir[i] = (i > 0) ? packet.ppg_ir[i-1] : 0;
                }
            }
            
            for (int i = 0; i < ECG_SAMPLES_PER_PACKET; i++) {
                ecg_sample_t ecg_sample;
                if (xQueueReceive(ecg_queue, &ecg_sample, pdMS_TO_TICKS(5)) == pdTRUE) {
                    packet.ecg_values[i] = ecg_sample.ecg_value;
                    packet.lo_status[i] = ecg_sample.lo_status;
                } else {
                    packet.ecg_values[i] = (i > 0) ? packet.ecg_values[i-1] : 0;
                    packet.lo_status[i] = (i > 0) ? packet.lo_status[i-1] : 0;
                }
            }
            
            if (xSemaphoreTake(system_data_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
                packet.battery_voltage = system_data.battery_voltage;
                packet.battery_soc = system_data.battery_soc;
                packet.temperature = system_data.temperature_valid ? system_data.temperature : -999.0f;
                xSemaphoreGive(system_data_mutex);
            } else {
                packet.battery_voltage = 0.0f;
                packet.battery_soc = 0.0f;
                packet.temperature = -999.0f;
            }
            
            packet.packet_end_timestamp = esp_timer_get_time();
            
#if OUTPUT_FORMAT == OUTPUT_FORMAT_PACKET
            print_packet_format(&packet);
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_APP
            print_app_format(&packet);
#endif
        }
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "NirogScan Optimized Starting...");
    ESP_LOGI(TAG, "PPG: %dHz, ECG: %dHz, Packet Rate: %.1fHz", 
             PPG_SAMPLE_RATE_HZ, ECG_SAMPLE_RATE_HZ, 
             1000000.0f / PACKET_TIMER_PERIOD_US);

    ppg_queue = xQueueCreate(QUEUE_SIZE, sizeof(ppg_sample_t));
    ecg_queue = xQueueCreate(QUEUE_SIZE, sizeof(ecg_sample_t));
    packet_queue = xQueueCreate(QUEUE_SIZE, sizeof(uint32_t));
    system_data_mutex = xSemaphoreCreateMutex();

    if (!ppg_queue || !ecg_queue || !packet_queue || !system_data_mutex) {
        ESP_LOGE(TAG, "Failed to create queues or mutex");
        return;
    }

    gpio_init_all();
    
    ESP_ERROR_CHECK(i2c_master_init());
    ESP_ERROR_CHECK(adc_init());
    ESP_ERROR_CHECK(max30101_init());

    ESP_LOGI(TAG, "Hardware initialized successfully");

#if OUTPUT_FORMAT == OUTPUT_FORMAT_PACKET
    printf("Type,Seq,PPG_Red[%d],PPG_IR[%d],ECG[%d],LO[%d],Battery_V,Battery_SOC,Temp,Time_Start,Time_End\n",
           PPG_SAMPLES_PER_PACKET, PPG_SAMPLES_PER_PACKET, 
           ECG_SAMPLES_PER_PACKET, ECG_SAMPLES_PER_PACKET);
#elif OUTPUT_FORMAT == OUTPUT_FORMAT_APP
    printf("# App Format: A1=PPG_Red[%d], A2=PPG_IR[%d], A3=ECG[%d], A4=LO_Status[%d], A5=Battery_V, A6=Battery_SOC, A7=Temp, A8=Seq, A9=Time_Start, A10=Time_End\n",
           PPG_SAMPLES_PER_PACKET, PPG_SAMPLES_PER_PACKET, 
           ECG_SAMPLES_PER_PACKET, ECG_SAMPLES_PER_PACKET);
#endif

    xTaskCreatePinnedToCore(packet_processing_task, "packet_proc", 8192, NULL, 5, NULL, 0);

    const esp_timer_create_args_t ppg_timer_args = {
        .callback = &ppg_timer_callback,
        .name = "ppg_timer"
    };
    const esp_timer_create_args_t ecg_timer_args = {
        .callback = &ecg_timer_callback,
        .name = "ecg_timer"
    };
    const esp_timer_create_args_t packet_timer_args = {
        .callback = &packet_timer_callback,
        .name = "packet_timer"
    };

    ESP_ERROR_CHECK(esp_timer_create(&ppg_timer_args, &ppg_timer));
    ESP_ERROR_CHECK(esp_timer_create(&ecg_timer_args, &ecg_timer));
    ESP_ERROR_CHECK(esp_timer_create(&packet_timer_args, &packet_timer));

    ESP_ERROR_CHECK(esp_timer_start_periodic(ppg_timer, PPG_TIMER_PERIOD_US));
    ESP_ERROR_CHECK(esp_timer_start_periodic(ecg_timer, ECG_TIMER_PERIOD_US));
    ESP_ERROR_CHECK(esp_timer_start_periodic(packet_timer, PACKET_TIMER_PERIOD_US));

    ESP_LOGI(TAG, "All timers started - system operational");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}