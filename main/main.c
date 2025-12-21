#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "driver/i2c.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_check.h"
#include <inttypes.h>

#define ENABLE_LOGS 1

static const char *TAG = "NirogScan";

#define I2C_MASTER_SCL_IO_0           9
#define I2C_MASTER_SDA_IO_0           8
#define I2C_MASTER_FREQ_HZ_0          400000
#define I2C_MASTER_NUM_0              I2C_NUM_0

#define I2C_MASTER_SCL_IO_1           11
#define I2C_MASTER_SDA_IO_1           10
#define I2C_MASTER_FREQ_HZ_1          400000
#define I2C_MASTER_NUM_1              I2C_NUM_1

#define MAX30102_I2C_ADDR             0x57
#define MAX17048_I2C_ADDR             0x36

#define ECG_ADC_CHANNEL               ADC_CHANNEL_1
#define ECG_ADC_ATTEN                 ADC_ATTEN_DB_12
#define ECG_GPIO_NUM                  GPIO_NUM_2

#define LO_PLUS_PIN                   GPIO_NUM_16
#define LO_MINUS_PIN                  GPIO_NUM_17

#define MAX30102_REG_FIFO_WR_PTR      0x04
#define MAX30102_REG_FIFO_OVF_COUNTER 0x05
#define MAX30102_REG_FIFO_RD_PTR      0x06
#define MAX30102_REG_FIFO_DATA        0x07
#define MAX30102_REG_FIFO_CONFIG      0x08
#define MAX30102_REG_MODE_CONFIG      0x09
#define MAX30102_REG_SPO2_CONFIG      0x0A
#define MAX30102_REG_LED1_PA          0x0C
#define MAX30102_REG_LED2_PA          0x0D
#define MAX30102_REG_PART_ID          0xFF

#define MAX17048_REG_VCELL            0x02
#define MAX17048_REG_SOC              0x04

#define SAMPLE_RATE_HZ                128
#define TIMER_PERIOD_US               (1000000 / SAMPLE_RATE_HZ)

#define PPG_READY_BIT                 BIT0
#define ECG_READY_BIT                 BIT1

#if ENABLE_LOGS
    #define LOG_I(tag, format, ...) ESP_LOGI(tag, format, ##__VA_ARGS__)
    #define LOG_E(tag, format, ...) ESP_LOGE(tag, format, ##__VA_ARGS__)
#else
    #define LOG_I(tag, format, ...)
    #define LOG_E(tag, format, ...)
#endif

typedef struct {
    uint32_t red;
    uint32_t ir;
    int ecg;
    float battery_voltage;
    float battery_percent;
    int lo_plus;
    int lo_minus;
    uint32_t timestamp;
} sensor_data_t;

static adc_oneshot_unit_handle_t adc1_handle;
static sensor_data_t sensor_data = {0};
static EventGroupHandle_t sync_event_group;
static esp_timer_handle_t periodic_timer;
static volatile uint32_t battery_counter = 0;

static esp_err_t i2c_master_init_0(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO_0,
        .scl_io_num = I2C_MASTER_SCL_IO_0,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ_0,
    };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM_0, &conf), TAG, "I2C0 config failed");
    return i2c_driver_install(I2C_MASTER_NUM_0, conf.mode, 0, 0, 0);
}

static esp_err_t i2c_master_init_1(void) {
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = I2C_MASTER_SDA_IO_1,
        .scl_io_num = I2C_MASTER_SCL_IO_1,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = I2C_MASTER_FREQ_HZ_1,
    };
    ESP_RETURN_ON_ERROR(i2c_param_config(I2C_MASTER_NUM_1, &conf), TAG, "I2C1 config failed");
    return i2c_driver_install(I2C_MASTER_NUM_1, conf.mode, 0, 0, 0);
}

static esp_err_t max30102_write_reg(uint8_t reg, uint8_t data) {
    uint8_t write_buf[2] = {reg, data};
    return i2c_master_write_to_device(I2C_MASTER_NUM_0, MAX30102_I2C_ADDR, write_buf, 2, pdMS_TO_TICKS(100));
}

static esp_err_t max30102_read_reg(uint8_t reg, uint8_t *data) {
    return i2c_master_write_read_device(I2C_MASTER_NUM_0, MAX30102_I2C_ADDR, &reg, 1, data, 1, pdMS_TO_TICKS(100));
}

static esp_err_t max30102_read_fifo(uint8_t *buffer, size_t len) {
    uint8_t reg = MAX30102_REG_FIFO_DATA;
    return i2c_master_write_read_device(I2C_MASTER_NUM_0, MAX30102_I2C_ADDR, &reg, 1, buffer, len, pdMS_TO_TICKS(100));
}

static esp_err_t max30102_init(void) {
    uint8_t part_id;
    ESP_RETURN_ON_ERROR(max30102_read_reg(MAX30102_REG_PART_ID, &part_id), TAG, "Part ID read failed");
    LOG_I(TAG, "MAX30102 Part ID: 0x%02X", part_id);

    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_MODE_CONFIG, 0x40), TAG, "Reset failed");
    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_FIFO_WR_PTR, 0x00), TAG, "WR PTR failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_FIFO_OVF_COUNTER, 0x00), TAG, "OVF failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_FIFO_RD_PTR, 0x00), TAG, "RD PTR failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_FIFO_CONFIG, 0x4F), TAG, "FIFO config failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_MODE_CONFIG, 0x03), TAG, "Mode config failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_SPO2_CONFIG, 0x27), TAG, "SpO2 config failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_LED1_PA, 0x24), TAG, "LED1 failed");
    ESP_RETURN_ON_ERROR(max30102_write_reg(MAX30102_REG_LED2_PA, 0x24), TAG, "LED2 failed");

    return ESP_OK;
}

static esp_err_t max17048_read_reg(uint8_t reg, uint16_t *data) {
    uint8_t buffer[2];
    esp_err_t ret = i2c_master_write_read_device(I2C_MASTER_NUM_1, MAX17048_I2C_ADDR, &reg, 1, buffer, 2, pdMS_TO_TICKS(100));
    if (ret == ESP_OK) {
        *data = (buffer[0] << 8) | buffer[1];
    }
    return ret;
}

static esp_err_t adc_init(void) {
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = ADC_UNIT_1,
    };
    ESP_RETURN_ON_ERROR(adc_oneshot_new_unit(&init_config, &adc1_handle), TAG, "ADC init failed");

    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_DEFAULT,
        .atten = ECG_ADC_ATTEN,
    };
    return adc_oneshot_config_channel(adc1_handle, ECG_ADC_CHANNEL, &config);
}

static void gpio_init_all(void) {
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << LO_PLUS_PIN) | (1ULL << LO_MINUS_PIN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    gpio_config(&io_conf);
}

static void IRAM_ATTR periodic_timer_callback(void* arg) {
    BaseType_t xHigherPriorityTaskWoken = pdFALSE;
    xEventGroupSetBitsFromISR(sync_event_group, PPG_READY_BIT | ECG_READY_BIT, &xHigherPriorityTaskWoken);
    if (xHigherPriorityTaskWoken) {
        portYIELD_FROM_ISR();
    }
}

static void ppg_task(void *pvParameters) {
    uint8_t fifo_data[6];

    while (1) {
        xEventGroupWaitBits(sync_event_group, PPG_READY_BIT, pdTRUE, pdFALSE, portMAX_DELAY);

        if (max30102_read_fifo(fifo_data, 6) == ESP_OK) {
            sensor_data.red = ((uint32_t)fifo_data[0] << 16) | ((uint32_t)fifo_data[1] << 8) | fifo_data[2];
            sensor_data.red &= 0x3FFFF;

            sensor_data.ir = ((uint32_t)fifo_data[3] << 16) | ((uint32_t)fifo_data[4] << 8) | fifo_data[5];
            sensor_data.ir &= 0x3FFFF;
        }
    }
}

static void ecg_task(void *pvParameters) {
    int adc_raw;
    uint16_t vcell, soc;

    while (1) {
        xEventGroupWaitBits(sync_event_group, ECG_READY_BIT, pdTRUE, pdFALSE, portMAX_DELAY);

        sensor_data.lo_plus = gpio_get_level(LO_PLUS_PIN);
        sensor_data.lo_minus = gpio_get_level(LO_MINUS_PIN);
        
        if (adc_oneshot_read(adc1_handle, ECG_ADC_CHANNEL, &adc_raw) == ESP_OK) {
            sensor_data.ecg = adc_raw;
        }

        battery_counter++;
        if (battery_counter >= 128) {
            battery_counter = 0;
            if (max17048_read_reg(MAX17048_REG_VCELL, &vcell) == ESP_OK &&
                max17048_read_reg(MAX17048_REG_SOC, &soc) == ESP_OK) {
                sensor_data.battery_voltage = (vcell >> 4) * 1.25 / 1000.0;
                sensor_data.battery_percent = (soc >> 8) + (soc & 0xFF) / 256.0;
            }
        }

        sensor_data.timestamp = esp_timer_get_time() / 1000;

        printf("%d,%lu,%lu,%.2f,%.1f,%d,%d\n",
            // (unsigned long)sensor_data.timestamp,
            sensor_data.ecg,
            (unsigned long)sensor_data.ir,
            (unsigned long)sensor_data.red,
            sensor_data.battery_voltage,
            sensor_data.battery_percent,
            sensor_data.lo_minus,
            sensor_data.lo_plus);
    }
}

void app_main(void) {
    LOG_I(TAG, "NirogScan Lite Starting...");
    LOG_I(TAG, "Sample Rate: %d Hz", SAMPLE_RATE_HZ);

    sync_event_group = xEventGroupCreate();
    if (sync_event_group == NULL) {
        LOG_E(TAG, "Failed to create event group");
        return;
    }

    gpio_init_all();
    vTaskDelay(pdMS_TO_TICKS(100));

    if (i2c_master_init_0() != ESP_OK) {
        LOG_E(TAG, "I2C0 init failed");
        return;
    }

    if (i2c_master_init_1() != ESP_OK) {
        LOG_E(TAG, "I2C1 init failed");
        return;
    }

    if (adc_init() != ESP_OK) {
        LOG_E(TAG, "ADC init failed");
        return;
    }

    if (max30102_init() != ESP_OK) {
        LOG_E(TAG, "MAX30102 init failed");
        return;
    }

    LOG_I(TAG, "All sensors initialized");

    printf("ecg,ir,red,battery_v,battery_percent,lo_minus,lo_plus\n");

    xTaskCreatePinnedToCore(ppg_task, "ppg_task", 4096, NULL, 10, NULL, 0);
    xTaskCreatePinnedToCore(ecg_task, "ecg_task", 4096, NULL, 10, NULL, 1);

    const esp_timer_create_args_t periodic_timer_args = {
        .callback = &periodic_timer_callback,
        .name = "periodic"
    };

    ESP_ERROR_CHECK(esp_timer_create(&periodic_timer_args, &periodic_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(periodic_timer, TIMER_PERIOD_US));

    LOG_I(TAG, "All tasks started, timer running at %d Hz", SAMPLE_RATE_HZ);
}