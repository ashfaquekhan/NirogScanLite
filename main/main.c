#include <stdio.h>
#include <string.h>
#include "esp_log.h"
#include "esp_vfs_fat.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tinyusb.h"
#include "tinyusb_default_config.h"
#include "tinyusb_msc.h"

static const char *TAG = "main";

#define BASE_PATH "/data"

void app_main(void)
{
    ESP_LOGI(TAG, "Initializing FatFS storage");

    static wl_handle_t wl_handle = WL_INVALID_HANDLE;
    
    const esp_partition_t *data_partition = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_DATA_FAT, NULL);
    
    if (data_partition == NULL) {
        ESP_LOGE(TAG, "Failed to find FAT partition");
        return;
    }

    esp_err_t err = wl_mount(data_partition, &wl_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to mount wear leveling");
        return;
    }

    tinyusb_msc_storage_config_t storage_cfg = {
        .mount_point = TINYUSB_MSC_STORAGE_MOUNT_APP,
        .medium.wl_handle = wl_handle,
        .fat_fs = {
            .base_path = BASE_PATH,
            .config.max_files = 5,
        },
    };

    tinyusb_msc_storage_handle_t storage_handle = NULL;
    err = tinyusb_msc_new_storage_spiflash(&storage_cfg, &storage_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create MSC storage");
        return;
    }

    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_LOGI(TAG, "Creating PDF file");
    FILE *f = fopen(BASE_PATH "/test.pdf", "wb");
    if (!f) {
        ESP_LOGE(TAG, "Failed to create PDF file");
        return;
    }

    fprintf(f, "%%PDF-1.4\n");
    fprintf(f, "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
    fprintf(f, "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
    fprintf(f, "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n");
    fprintf(f, "4 0 obj\n<< /Length 44 >>\nstream\nBT /F1 12 Tf 50 750 Td (Test!) Tj ET\nendstream\nendobj\n");
    fprintf(f, "xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000207 00000 n \ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n300\n%%%%EOF\n");

    fclose(f);
    ESP_LOGI(TAG, "PDF created successfully");

    vTaskDelay(pdMS_TO_TICKS(100));

    ESP_LOGI(TAG, "Starting USB MSC");
    const tinyusb_config_t tusb_cfg = TINYUSB_DEFAULT_CONFIG();
    err = tinyusb_driver_install(&tusb_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to install TinyUSB");
        return;
    }

    ESP_LOGI(TAG, "SUCCESS! PDF created. Connect USB to access as storage.");
}